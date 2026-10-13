// mask-check: quick visual comparison of Apple Vision masks vs SAM 2.1 (Core ML) on Sony RAWs.
// Build: swiftc -O -swift-version 5 main.swift -o maskcheck
// Usage: maskcheck --list images.tsv --out DIR --models DIR [--only decode|person|subject|saliency|sam] [--units all|cpuAndGPU|cpuOnly]
import AppKit
import CoreImage
import CoreML
import Foundation
import Vision

// MARK: - args

var argv = Array(CommandLine.arguments.dropFirst())
func arg(_ name: String) -> String? {
    guard let i = argv.firstIndex(of: name), i + 1 < argv.count else { return nil }
    return argv[i + 1]
}
guard let listPath = arg("--list"), let outDir = arg("--out"), let modelsDir = arg("--models") else {
    FileHandle.standardError.write("usage: maskcheck --list images.tsv --out DIR --models DIR [--only MODE] [--units U]\n".data(using: .utf8)!)
    exit(2)
}
let only = arg("--only")  // memory-measurement mode: run a single method, write no images
let unitsName = arg("--units") ?? "all"
let warmReps = 3
let longEdge: CGFloat = 2048
let panelEdge: CGFloat = 800

func log(_ s: String) { FileHandle.standardError.write((s + "\n").data(using: .utf8)!) }
func ms(_ block: () throws -> Void) rethrows -> Double {
    let t = DispatchTime.now().uptimeNanoseconds
    try block()
    return Double(DispatchTime.now().uptimeNanoseconds - t) / 1e6
}
func median(_ v: [Double]) -> Double {
    let s = v.sorted()
    return s.isEmpty ? .nan : (s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2)
}

// MARK: - memory

func peakRSSMB() -> Double {
    var ru = rusage()
    getrusage(RUSAGE_SELF, &ru)
    return Double(ru.ru_maxrss) / 1048576  // bytes on macOS
}
func footprintMB() -> (now: Double, peak: Double) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return (.nan, .nan) }
    return (Double(info.phys_footprint) / 1048576, Double(info.ledger_phys_footprint_peak) / 1048576)
}

// MARK: - decode

let ciContext = CIContext(options: [.workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!])
let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

/// CIRAWFilter decode to ~2048 px long edge; falls back to the embedded preview.
func decode(_ url: URL) -> (CGImage, String)? {
    if let raw = CIRAWFilter(imageURL: url) {
        let native = raw.nativeSize
        let m = max(native.width, native.height)
        if m > 0 { raw.scaleFactor = Float(min(1, longEdge / m)) }
        if let out = raw.outputImage {
            // scaleFactor snaps to demosaic-friendly sizes; finish with an exact resample
            let s = longEdge / max(out.extent.width, out.extent.height)
            let scaled = s < 0.999 ? out.transformed(by: CGAffineTransform(scaleX: s, y: s)) : out
            if let cg = ciContext.createCGImage(scaled, from: scaled.extent.integral, format: .RGBA8, colorSpace: sRGB) {
                return (cg, "raw")
            }
        }
    }
    let opts: [CFString: Any] = [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: longEdge,
    ]
    if let src = CGImageSourceCreateWithURL(url as CFURL, nil),
        let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary)
    {
        return (cg, "preview")
    }
    return nil
}

// MARK: - masks

/// A soft mask (0...1) at arbitrary resolution, top-left origin.
struct Mask {
    var w: Int, h: Int
    var v: [Float]
    var coverage: Double { v.isEmpty ? 0 : Double(v.reduce(0) { $0 + ($1 > 0.5 ? 1 : 0) }) / Double(v.count) }
    func ciImage(size: CGSize) -> CIImage {
        let bytes = v.map { UInt8(max(0, min(1, $0)) * 255) }
        let cg = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil, shouldInterpolate: true,
            intent: .defaultIntent)!
        return CIImage(cgImage: cg).transformed(
            by: CGAffineTransform(scaleX: size.width / CGFloat(w), y: size.height / CGFloat(h)))
    }
}

func mask(from pb: CVPixelBuffer) -> Mask {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), rb = CVPixelBufferGetBytesPerRow(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!
    var v = [Float](repeating: 0, count: w * h)
    switch CVPixelBufferGetPixelFormatType(pb) {
    case kCVPixelFormatType_OneComponent8:
        for y in 0..<h {
            let row = base.advanced(by: y * rb).assumingMemoryBound(to: UInt8.self)
            for x in 0..<w { v[y * w + x] = Float(row[x]) / 255 }
        }
    case kCVPixelFormatType_OneComponent32Float:
        for y in 0..<h {
            let row = base.advanced(by: y * rb).assumingMemoryBound(to: Float.self)
            for x in 0..<w { v[y * w + x] = row[x] }
        }
    default:
        log("unexpected mask pixel format \(CVPixelBufferGetPixelFormatType(pb))")
    }
    return Mask(w: w, h: h, v: v)
}

func visionPerson(_ cg: CGImage) throws -> Mask? {
    let req = VNGeneratePersonSegmentationRequest()
    req.qualityLevel = .accurate
    req.outputPixelFormat = kCVPixelFormatType_OneComponent8
    try VNImageRequestHandler(cgImage: cg).perform([req])
    return req.results?.first.map { mask(from: $0.pixelBuffer) }
}

func visionSubject(_ cg: CGImage) throws -> (Mask, Int)? {
    let req = VNGenerateForegroundInstanceMaskRequest()
    let handler = VNImageRequestHandler(cgImage: cg)
    try handler.perform([req])
    guard let obs = req.results?.first else { return nil }
    let pb = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: handler)
    return (mask(from: pb), obs.allInstances.count)
}

/// Attention saliency heatmap plus its (3x3-smoothed) peak in normalised top-left coords.
func visionSaliency(_ cg: CGImage) throws -> (Mask, CGPoint)? {
    let req = VNGenerateAttentionBasedSaliencyImageRequest()
    try VNImageRequestHandler(cgImage: cg).perform([req])
    guard let obs = req.results?.first else { return nil }
    let m = mask(from: obs.pixelBuffer)
    var best: Float = -1, bx = m.w / 2, by = m.h / 2
    for y in 0..<m.h {
        for x in 0..<m.w {
            var s: Float = 0
            for dy in -1...1 {
                for dx in -1...1 {
                    let xx = min(max(x + dx, 0), m.w - 1), yy = min(max(y + dy, 0), m.h - 1)
                    s += m.v[yy * m.w + xx]
                }
            }
            if s > best { best = s; bx = x; by = y }
        }
    }
    return (m, CGPoint(x: (CGFloat(bx) + 0.5) / CGFloat(m.w), y: (CGFloat(by) + 0.5) / CGFloat(m.h)))
}

/// Centre of the largest detected face, normalised top-left coords.
func visionFace(_ cg: CGImage) throws -> CGPoint? {
    let req = VNDetectFaceRectanglesRequest()
    try VNImageRequestHandler(cgImage: cg).perform([req])
    guard let f = req.results?.max(by: { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height })
    else { return nil }
    return CGPoint(x: f.boundingBox.midX, y: 1 - f.boundingBox.midY)
}

// MARK: - SAM 2.1 (Core ML, apple/coreml-sam2.1-small)

final class SAM2 {
    let encoder: MLModel, prompt: MLModel, decoder: MLModel
    var loadMS: [String: Double] = [:]
    init(dir: URL, units: MLComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        var times: [String: Double] = [:]
        func load(_ name: String) throws -> MLModel {
            var m: MLModel!
            let url = dir.appendingPathComponent("SAM2_1Small\(name)FLOAT16.mlmodelc")
            times[name] = try ms { m = try MLModel(contentsOf: url, configuration: cfg) }
            return m
        }
        encoder = try load("ImageEncoder")
        prompt = try load("PromptEncoder")
        decoder = try load("MaskDecoder")
        loadMS = times
    }

    func encode(_ cg: CGImage) throws -> MLFeatureProvider {
        let c = encoder.modelDescription.inputDescriptionsByName["image"]!.imageConstraint!
        let fv = try MLFeatureValue(
            cgImage: cg, constraint: c, options: [.cropAndScale: VNImageCropAndScaleOption.scaleFill.rawValue])
        return try encoder.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": fv]))
    }

    /// One positive point (normalised top-left coords) -> all three candidate masks with predicted IoU.
    func segment(_ emb: MLFeatureProvider, point: CGPoint) throws -> [(Mask, Float)] {
        let pts = try MLMultiArray(shape: [1, 1, 2], dataType: .float16)
        pts[[0, 0, 0]] = NSNumber(value: Float(point.x) * 1024)
        pts[[0, 0, 1]] = NSNumber(value: Float(point.y) * 1024)
        let labels = try MLMultiArray(shape: [1, 1], dataType: .float16)
        labels[[0, 0]] = 1
        let p = try prompt.prediction(from: MLDictionaryFeatureProvider(dictionary: ["points": pts, "labels": labels]))
        let out = try decoder.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "image_embedding": emb.featureValue(for: "image_embedding")!,
                "sparse_embedding": p.featureValue(for: "sparse_embeddings")!,
                "dense_embedding": p.featureValue(for: "dense_embeddings")!,
                "feats_s0": emb.featureValue(for: "feats_s0")!,
                "feats_s1": emb.featureValue(for: "feats_s1")!,
            ]))
        let scores = out.featureValue(for: "scores")!.multiArrayValue!
        let masks = out.featureValue(for: "low_res_masks")!.multiArrayValue!
        let n = 256 * 256
        let logits = masks.withUnsafeBufferPointer(ofType: Float16.self) { Array($0) }
        return (0..<3).map { k in
            let v = (0..<n).map { 1 / (1 + exp(-Float(logits[k * n + $0]))) }  // logits -> soft mask
            return (Mask(w: 256, h: 256, v: v), scores[k].floatValue)
        }
    }
}

// MARK: - compositing

func panel(_ base: CIImage, mask: Mask?, tint: CIColor, title: String, note: String, dot: CGPoint?, size: CGSize) -> CGImage {
    var img = base
    if let mask {
        let dim = base.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.15, kCIInputBrightnessKey: -0.05])
            .applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: -2.0])
        let tinted = CIImage(color: tint).cropped(to: base.extent)
            .applyingFilter("CISourceOverCompositing", parameters: [kCIInputBackgroundImageKey: base])
        img = tinted.applyingFilter(
            "CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: dim, kCIInputMaskImageKey: mask.ciImage(size: base.extent.size)])
    }
    let cg = ciContext.createCGImage(img, from: base.extent, format: .RGBA8, colorSpace: sRGB)!
    let ctx = CGContext(
        data: nil, width: Int(size.width), height: Int(size.height), bitsPerComponent: 8, bytesPerRow: 0, space: sRGB,
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.draw(cg, in: CGRect(origin: .zero, size: size))
    if let dot {
        let c = CGPoint(x: dot.x * size.width, y: (1 - dot.y) * size.height)
        ctx.setStrokeColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1)); ctx.setLineWidth(5)
        ctx.strokeEllipse(in: CGRect(x: c.x - 9, y: c.y - 9, width: 18, height: 18))
        ctx.setStrokeColor(CGColor(red: 1, green: 1, blue: 0, alpha: 1)); ctx.setLineWidth(2.5)
        ctx.strokeEllipse(in: CGRect(x: c.x - 9, y: c.y - 9, width: 18, height: 18))
    }
    ctx.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 0.62))
    ctx.fill(CGRect(x: 0, y: size.height - 30, width: size.width, height: 30))
    let g = NSGraphicsContext(cgContext: ctx, flipped: false)
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = g
    (title as NSString).draw(
        at: CGPoint(x: 8, y: size.height - 25),
        withAttributes: [.font: NSFont.boldSystemFont(ofSize: 16), .foregroundColor: NSColor.white])
    let attrs: [NSAttributedString.Key: Any] = [.font: NSFont.monospacedDigitSystemFont(ofSize: 14, weight: .regular), .foregroundColor: NSColor(white: 0.85, alpha: 1)]
    let nw = (note as NSString).size(withAttributes: attrs).width
    (note as NSString).draw(at: CGPoint(x: size.width - nw - 8, y: size.height - 24), withAttributes: attrs)
    NSGraphicsContext.restoreGraphicsState()
    return ctx.makeImage()!
}

func writeJPEG(_ cg: CGImage, to url: URL, quality: Double) {
    let d = CGImageDestinationCreateWithURL(url as CFURL, "public.jpeg" as CFString, 1, nil)!
    CGImageDestinationAddImage(d, cg, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
    CGImageDestinationFinalize(d)
}

// MARK: - run

let units: MLComputeUnits = ["cpuAndGPU": .cpuAndGPU, "cpuOnly": .cpuOnly, "cpuAndNeuralEngine": .cpuAndNeuralEngine][unitsName] ?? .all
let rows = try String(contentsOfFile: listPath, encoding: .utf8).split(separator: "\n").map { $0.split(separator: "\t").map(String.init) }
let outURL = URL(fileURLWithPath: outDir)
try FileManager.default.createDirectory(at: outURL.appendingPathComponent("sheets"), withIntermediateDirectories: true)

func want(_ m: String) -> Bool { only == nil || only == m }
var sam: SAM2?
if want("sam") {
    sam = try SAM2(dir: URL(fileURLWithPath: modelsDir), units: units)
    log("sam2 load ms (units=\(unitsName)): \(sam!.loadMS)")
}

var csv = [
    "id,file,kind,width,height,decode,decode_ms,prompt_src,prompt_x,prompt_y,person_ms,person_cov,subject_ms,subject_instances,subject_cov,saliency_ms,face_ms,sam_encode_ms,sam_prompt_decode_ms,sam_total_ms,sam_score,sam_cov,sam_largest_score,sam_largest_cov,footprint_mb"
]
var cold: [String: Double] = [:]
func f1(_ d: Double) -> String { d.isNaN ? "" : String(format: "%.1f", d) }
func f3(_ d: Double) -> String { d.isNaN ? "" : String(format: "%.3f", d) }

for (i, row) in rows.enumerated() where row.count >= 2 {
    try autoreleasepool {
        let kind = row[0], url = URL(fileURLWithPath: row[1])
        let id = String(format: "%02d_%@_%@", i + 1, kind, url.deletingPathExtension().lastPathComponent)
        var dec: (CGImage, String)?
        let decodeMS = ms { dec = decode(url) }
        guard let (cg, how) = dec else { log("\(id): decode failed"); return }
        let size = CGSize(width: cg.width, height: cg.height)

        // Each method: one first call (recorded as cold on image 0), then median of warmReps.
        func timed(_ name: String, _ body: () throws -> Void) rethrows -> Double {
            let first = try ms(body)
            if i == 0 { cold[name] = first }
            if only != nil { return first }
            return median(try (0..<warmReps).map { _ in try ms(body) })
        }

        var person: Mask?, subject: (Mask, Int)?, sal: (Mask, CGPoint)?, face: CGPoint?
        var tPerson = Double.nan, tSubject = Double.nan, tSal = Double.nan, tFace = Double.nan
        if want("person") { tPerson = try timed("person") { person = try visionPerson(cg) } }
        if want("subject") { tSubject = try timed("subject") { subject = try visionSubject(cg) } }
        if want("saliency") || want("sam") {
            tSal = try timed("saliency") { sal = try visionSaliency(cg) }
            tFace = try timed("face") { face = try visionFace(cg) }
        }
        let point = face ?? sal?.1 ?? CGPoint(x: 0.5, y: 0.5)
        let src = face != nil ? "face" : "saliency"

        var samAll: [(Mask, Float)] = []
        var tEnc = Double.nan, tDec = Double.nan
        if let sam {
            var emb: MLFeatureProvider!
            tEnc = try timed("sam_encode") { emb = try sam.encode(cg) }
            tDec = try timed("sam_prompt_decode") { samAll = try sam.segment(emb, point: point) }
        }
        // default SAM behaviour = highest predicted IoU; also keep the largest candidate (the "whole object" guess)
        let samOut = samAll.max { $0.1 < $1.1 }
        let samBig = samAll.max { $0.0.coverage < $1.0.coverage }

        let fp = footprintMB()
        csv.append(
            [
                id, url.lastPathComponent, kind, "\(cg.width)", "\(cg.height)", how, f1(decodeMS), src, f3(point.x), f3(point.y),
                f1(tPerson), f3(person?.coverage ?? .nan), f1(tSubject), subject.map { "\($0.1)" } ?? "", f3(subject?.0.coverage ?? .nan),
                f1(tSal), f1(tFace), f1(tEnc), f1(tDec), f1(tEnc + tDec), samOut.map { f3(Double($0.1)) } ?? "",
                f3(samOut?.0.coverage ?? .nan), samBig.map { f3(Double($0.1)) } ?? "", f3(samBig?.0.coverage ?? .nan), f1(fp.now),
            ].joined(separator: ","))
        log("\(id) \(how) person=\(f1(tPerson)) subject=\(f1(tSubject)) sal=\(f1(tSal)) sam=\(f1(tEnc))+\(f1(tDec)) fp=\(f1(fp.now))MB")
        if only != nil { return }

        // 3x2 sheet: original | Vision person | Vision subject / saliency | SAM 2.1 best-score | SAM 2.1 largest
        let s = panelEdge / max(size.width, size.height)
        let ps = CGSize(width: (size.width * s).rounded(), height: (size.height * s).rounded())
        let base = CIImage(cgImage: cg)
        let empty = Mask(w: 1, h: 1, v: [0])
        let pm = person.map { $0.coverage > 0.002 ? $0 : empty } ?? empty
        let panels = [
            panel(base, mask: nil, tint: .clear, title: "Original  (o = \(src) prompt)", note: id, dot: point, size: ps),
            panel(base, mask: pm, tint: CIColor(red: 0.1, green: 0.5, blue: 1, alpha: 0.28), title: "Vision person (accurate)",
                note: "\(f1(tPerson)) ms  \(Int((person?.coverage ?? 0) * 100))%", dot: nil, size: ps),
            panel(base, mask: subject?.0 ?? empty, tint: CIColor(red: 0.1, green: 0.9, blue: 0.3, alpha: 0.28), title: "Vision subject (\(subject?.1 ?? 0) inst.)",
                note: "\(f1(tSubject)) ms  \(Int((subject?.0.coverage ?? 0) * 100))%", dot: nil, size: ps),
            panel(base, mask: sal?.0 ?? empty, tint: CIColor(red: 1, green: 0.6, blue: 0, alpha: 0.28), title: "Vision attention saliency",
                note: "\(f1(tSal)) ms", dot: point, size: ps),
            panel(base, mask: samOut?.0 ?? empty, tint: CIColor(red: 1, green: 0.3, blue: 0.7, alpha: 0.28), title: "SAM 2.1 small: best score",
                note: "\(f1(tEnc + tDec)) ms  \(Int((samOut?.0.coverage ?? 0) * 100))%  iou \(samOut.map { f3(Double($0.1)) } ?? "-")", dot: point, size: ps),
            panel(base, mask: samBig?.0 ?? empty, tint: CIColor(red: 1, green: 0.3, blue: 0.7, alpha: 0.28), title: "SAM 2.1 small: largest of 3",
                note: "\(Int((samBig?.0.coverage ?? 0) * 100))%  iou \(samBig.map { f3(Double($0.1)) } ?? "-")", dot: point, size: ps),
        ]
        let gap: CGFloat = 6
        let sheet = CGContext(
            data: nil, width: Int(ps.width * 3 + gap * 2), height: Int(ps.height * 2 + gap), bitsPerComponent: 8, bytesPerRow: 0,
            space: sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        sheet.setFillColor(CGColor(gray: 0.08, alpha: 1))
        sheet.fill(CGRect(x: 0, y: 0, width: sheet.width, height: sheet.height))
        for (k, p) in panels.enumerated() {
            sheet.draw(p, in: CGRect(x: CGFloat(k % 3) * (ps.width + gap), y: CGFloat(1 - k / 3) * (ps.height + gap), width: ps.width, height: ps.height))
        }
        writeJPEG(sheet.makeImage()!, to: outURL.appendingPathComponent("sheets/\(id).jpg"), quality: 0.78)
    }
}

let fp = footprintMB()
var summary: [String: Any] = [
    "mode": only ?? "all", "units": unitsName, "images": rows.count, "peak_rss_mb": peakRSSMB(),
    "peak_footprint_mb": fp.peak, "cold_first_call_ms": cold,
]
if let sam { summary["sam_load_ms"] = sam.loadMS }
let tag = (only ?? "all") + (unitsName == "all" ? "" : "_" + unitsName)
try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
    .write(to: outURL.appendingPathComponent("run_\(tag).json"))
try csv.joined(separator: "\n").appending("\n").write(
    to: outURL.appendingPathComponent(tag == "all" ? "results.csv" : "run_\(tag).csv"), atomically: true, encoding: .utf8)
log("done mode=\(tag) peakRSS=\(f1(peakRSSMB()))MB peakFootprint=\(f1(fp.peak))MB")
