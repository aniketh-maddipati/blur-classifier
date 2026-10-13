// stitch-check: panorama stitching experiment for Lumina.
// RAW decode (Core Image) -> pairwise registration (Vision) -> planar warp + feather blend (Metal).
// Reads RAWs in place, never writes next to them.
//
// Build:  swiftc -O -swift-version 5 main.swift -o stitch-check
// Run:    ./stitch-check --out out --scale 0.5 a.ARW b.ARW c.ARW

import CoreImage
import Foundation
import ImageIO
import Metal
import Vision
import simd

// MARK: - Options

struct Options {
    var inputs: [URL] = []
    var out = URL(fileURLWithPath: "out")
    var name = "pano"
    var scale = 0.5          // CIRAWFilter scaleFactor for the working size
    var regEdge = 2048       // long edge of the images handed to Vision
    var power: Float = 2     // feather weight exponent; higher = narrower transition
    var gain = true          // per-frame exposure gain from overlap means
    var lockWB = true        // use the first frame's white balance for all frames
    var crops = 3
    var solo = -1            // debug: draw only this frame, to compare a seam against a single exposure
    var cropSize = (w: 1600, h: 1000)
}

func parseOptions() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    func next() -> String {
        guard !args.isEmpty else { fatalError("missing value for option") }
        return args.removeFirst()
    }
    while !args.isEmpty {
        let a = args.removeFirst()
        switch a {
        case "--out": o.out = URL(fileURLWithPath: next())
        case "--name": o.name = next()
        case "--scale": o.scale = Double(next())!
        case "--reg-edge": o.regEdge = Int(next())!
        case "--power": o.power = Float(next())!
        case "--no-gain": o.gain = false
        case "--no-lock-wb": o.lockWB = false
        case "--crops": o.crops = Int(next())!
        case "--solo": o.solo = Int(next())!
        default: o.inputs.append(URL(fileURLWithPath: a))
        }
    }
    guard o.inputs.count >= 2 else {
        print("usage: stitch-check [--out dir] [--name pano] [--scale 0.5] [--reg-edge 2048] [--power 2] [--no-gain] [--no-lock-wb] raw1 raw2 ...")
        exit(2)
    }
    return o
}

// MARK: - Small helpers

typealias M3 = simd_double3x3

func m3(_ r0: [Double], _ r1: [Double], _ r2: [Double]) -> M3 {
    M3(rows: [SIMD3(r0[0], r0[1], r0[2]), SIMD3(r1[0], r1[1], r1[2]), SIMD3(r2[0], r2[1], r2[2])])
}

func normalized(_ m: M3) -> M3 {
    let k = m[2][2]
    return abs(k) > 1e-12 ? m * (1 / k) : m
}

func apply(_ m: M3, _ x: Double, _ y: Double) -> (x: Double, y: Double, z: Double) {
    let v = m * SIMD3(x, y, 1)
    return (v.x / v.z, v.y / v.z, v.z)
}

func now() -> Double { CFAbsoluteTimeGetCurrent() }

func peakFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? UInt64(info.ledger_phys_footprint_peak) : 0
}

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    var buf = [CChar](repeating: 0, count: max(size, 1))
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(cString: buf)
}

// MARK: - Grayscale buffers and overlap scoring

struct Gray {
    let w: Int, h: Int
    let px: [Float]   // sRGB-encoded luma 0...1, row 0 = top
    let hp: [Float]   // px minus a local mean: detail only, so flat sky and smooth gradients cannot fake a match

    func sample(_ px: [Float], _ x: Double, _ y: Double) -> Float? {
        // Coordinates have pixel centres at +0.5.
        let fx = x - 0.5, fy = y - 0.5
        guard fx >= 0, fy >= 0, fx < Double(w - 1), fy < Double(h - 1) else { return nil }
        let x0 = Int(fx), y0 = Int(fy)
        let tx = Float(fx - Double(x0)), ty = Float(fy - Double(y0))
        let i = y0 * w + x0
        let a = px[i] + (px[i + 1] - px[i]) * tx
        let b = px[i + w] + (px[i + w + 1] - px[i + w]) * tx
        return a + (b - a) * ty
    }
}

func makeGray(_ cg: CGImage) -> Gray {
    let w = cg.width, h = cg.height
    var bytes = [UInt8](repeating: 0, count: w * h)
    bytes.withUnsafeMutableBytes { p in
        let ctx = CGContext(data: p.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w,
                            space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    let px = bytes.map { Float($0) / 255 }
    return Gray(w: w, h: h, px: px, hp: highPass(px, w: w, h: h, radius: 6))
}

func highPass(_ px: [Float], w: Int, h: Int, radius r: Int) -> [Float] {
    // Summed-area table box blur.
    var sat = [Double](repeating: 0, count: (w + 1) * (h + 1))
    for y in 0..<h {
        var row = 0.0
        for x in 0..<w {
            row += Double(px[y * w + x])
            sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row
        }
    }
    var out = [Float](repeating: 0, count: w * h)
    for y in 0..<h {
        let y0 = max(0, y - r), y1 = min(h, y + r + 1)
        for x in 0..<w {
            let x0 = max(0, x - r), x1 = min(w, x + r + 1)
            let sum = sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0] + sat[y0 * (w + 1) + x0]
            out[y * w + x] = px[y * w + x] - Float(sum / Double((x1 - x0) * (y1 - y0)))
        }
    }
    return out
}

struct OverlapScore {
    var ncc = -1.0       // normalised cross-correlation of high-passed luma inside the overlap
    var overlap = 0.0    // fraction of the reference frame covered by the floating frame
    var mad = 1.0        // mean absolute luma difference (0...1, sRGB-encoded)
    var linRatio = 1.0   // mean linear luma of reference / floating inside the overlap
}

/// `h` maps floating-frame pixels to reference-frame pixels.
func score(ref: Gray, flt: Gray, h: M3, stride step: Int = 3) -> OverlapScore {
    let inv = h.inverse
    var n = 0.0, total = 0.0
    var sa = 0.0, sb = 0.0, saa = 0.0, sbb = 0.0, sab = 0.0, sad = 0.0, la = 0.0, lb = 0.0
    var y = 0
    while y < ref.h {
        var x = 0
        while x < ref.w {
            total += 1
            let q = inv * SIMD3(Double(x) + 0.5, Double(y) + 0.5, 1)
            if q.z > 1e-9, let b = flt.sample(flt.px, q.x / q.z, q.y / q.z), let hb = flt.sample(flt.hp, q.x / q.z, q.y / q.z) {
                let da = Double(ref.px[y * ref.w + x]), db = Double(b)
                let ha = Double(ref.hp[y * ref.w + x]), hb = Double(hb)
                n += 1; sa += ha; sb += hb; saa += ha * ha; sbb += hb * hb; sab += ha * hb
                sad += abs(da - db); la += pow(da, 2.2); lb += pow(db, 2.2)
            }
            x += step
        }
        y += step
    }
    var s = OverlapScore()
    guard n > 50 else { return s }
    let va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n)
    s.overlap = n / total
    s.mad = sad / n
    s.linRatio = lb > 0 ? la / lb : 1
    if va > 1e-10, vb > 1e-10 { s.ncc = (sab / n - (sa / n) * (sb / n)) / (va * vb).squareRoot() }
    return s
}

/// Rejects transforms that fold, mirror or wildly rescale the frame.
func isSane(_ h: M3, w: Double, h hh: Double) -> Bool {
    let c = [apply(h, 0, 0), apply(h, w, 0), apply(h, w, hh), apply(h, 0, hh)]
    guard c.allSatisfy({ $0.z > 1e-6 && $0.x.isFinite && $0.y.isFinite }) else { return false }
    var area = 0.0
    var signs = Set<Bool>()
    for i in 0..<4 {
        let a = c[i], b = c[(i + 1) % 4], d = c[(i + 2) % 4]
        area += a.x * b.y - b.x * a.y
        signs.insert((b.x - a.x) * (d.y - b.y) - (b.y - a.y) * (d.x - b.x) > 0)
    }
    let ratio = (area / 2) / (w * hh)   // positive = orientation preserved (y down)
    return signs.count == 1 && ratio > 0.4 && ratio < 2.5
}

// MARK: - Vision registration

struct PairResult {
    var h = matrix_identity_double3x3   // floating (j) -> reference (i), registration-size pixels, top-left origin
    var score = OverlapScore()
    var method = "none"
    var convention = -1
    var ok = false
}

/// Vision does not document the coordinate convention of its transforms in a way worth trusting blind,
/// so try the plausible readings and keep whichever actually lines the two images up.
func candidates(_ m: M3, height: Double) -> [M3] {
    let flip = m3([1, 0, 0], [0, -1, height], [0, 0, 1])
    var out: [M3] = []
    for base in [m, m.transpose] {
        for c in [base, base.inverse] {
            out.append(normalized(c))
            out.append(normalized(flip * c * flip))
        }
    }
    return out
}

var visionSeconds = 0.0, coarseSeconds = 0.0

/// Runs both Vision requests on a pair and returns every plausible reading of the results,
/// in the pixel coordinates of the images given (top-left origin).
func visionCandidates(refCG: CGImage, fltCG: CGImage) -> [(M3, String)] {
    var out: [(M3, String)] = []
    let t0 = now()
    defer { visionSeconds += now() - t0 }
    let handler = VNImageRequestHandler(cgImage: refCG, options: [:])
    let hh = Double(refCG.height)

    let hreq = VNHomographicImageRegistrationRequest(targetedCGImage: fltCG, options: [:])
    if (try? handler.perform([hreq])) != nil,
       let obs = hreq.results?.first as? VNImageHomographicAlignmentObservation {
        let w = obs.warpTransform
        let m = M3(columns: (SIMD3<Double>(w.columns.0), SIMD3<Double>(w.columns.1), SIMD3<Double>(w.columns.2)))
        out += candidates(m, height: hh).map { ($0, "homographic") }
    }
    let treq = VNTranslationalImageRegistrationRequest(targetedCGImage: fltCG, options: [:])
    if (try? handler.perform([treq])) != nil,
       let obs = treq.results?.first as? VNImageTranslationAlignmentObservation {
        let t = obs.alignmentTransform
        let m = m3([Double(t.a), Double(t.c), Double(t.tx)], [Double(t.b), Double(t.d), Double(t.ty)], [0, 0, 1])
        out += candidates(m, height: hh).prefix(4).map { ($0, "translational") }
    }
    return out
}

func shrink(_ g: Gray, width: Int) -> Gray {
    let k = max(1, g.w / width)
    let w = g.w / k, h = g.h / k
    var px = [Float](repeating: 0, count: w * h)
    for y in 0..<h {
        for x in 0..<w {
            var sum: Float = 0
            for yy in 0..<k { for xx in 0..<k { sum += g.px[(y * k + yy) * g.w + x * k + xx] } }
            px[y * w + x] = sum / Float(k * k)
        }
    }
    return Gray(w: w, h: h, px: px, hp: highPass(px, w: w, h: h, radius: 3))
}

/// Exhaustive search for the integer shift (floating -> reference) that best correlates two thumbnails.
/// Vision's registration only converges when frames nearly coincide, so a pan needs this head start.
func coarseShift(ref: Gray, flt: Gray, minOverlap: Double = 0.12) -> (dx: Int, dy: Int, ncc: Double)? {
    let w = ref.w, h = ref.h
    var bestOne: (Int, Int, Double)?
    for dy in -(h - 1)...(h - 1) {
        let y0 = max(0, dy), y1 = min(h, h + dy)
        for dx in -(w - 1)...(w - 1) {
            let x0 = max(0, dx), x1 = min(w, w + dx)
            let count = (x1 - x0) * (y1 - y0)
            guard Double(count) >= minOverlap * Double(w * h) else { continue }
            var sa: Float = 0, sb: Float = 0, saa: Float = 0, sbb: Float = 0, sab: Float = 0
            for y in y0..<y1 {
                let ra = y * w, rb = (y - dy) * w - dx
                for x in x0..<x1 {
                    let a = ref.hp[ra + x], b = flt.hp[rb + x]
                    sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b
                }
            }
            let n = Float(count)
            let va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n)
            guard va > 1e-9, vb > 1e-9 else { continue }
            let c = Double((sab / n - (sa / n) * (sb / n)) / (va * vb).squareRoot())
            if c > (bestOne?.2 ?? -2) { bestOne = (dx, dy, c) }
        }
    }
    return bestOne.map { (dx: $0.0, dy: $0.1, ncc: $0.2) }
}

func translation(_ x: Double, _ y: Double) -> M3 { m3([1, 0, x], [0, 1, y], [0, 0, 1]) }

func register(refCG: CGImage, fltCG: CGImage, ref: Gray, flt: Gray, refSmall: Gray, fltSmall: Gray, minNCC: Double) -> PairResult {
    // 1. Vision on the whole frames (enough when the frames nearly coincide).
    var cands = visionCandidates(refCG: refCG, fltCG: fltCG)

    // 2. Coarse shift, then Vision on just the overlapping crops.
    let t0 = now()
    let coarse = coarseShift(ref: refSmall, flt: fltSmall)
    coarseSeconds += now() - t0
    if let c = coarse {
        let k = Double(ref.w) / Double(refSmall.w)
        let dx = (Double(c.dx) * k).rounded(), dy = (Double(c.dy) * k).rounded()
        cands.append((translation(dx, dy), "coarse shift only"))
        let frame = CGRect(x: 0, y: 0, width: ref.w, height: ref.h)
        let inRef = frame.intersection(frame.offsetBy(dx: dx, dy: dy))
        let inFlt = inRef.offsetBy(dx: -dx, dy: -dy)
        if inRef.width >= 64, inRef.height >= 64, let cr = refCG.cropping(to: inRef), let cf = fltCG.cropping(to: inFlt) {
            let pre = translation(-inFlt.minX, -inFlt.minY), post = translation(inRef.minX, inRef.minY)
            cands += visionCandidates(refCG: cr, fltCG: cf).map { (normalized(post * $0.0 * pre), "coarse shift + " + $0.1) }
        }
    }

    var r = PairResult()
    for (k, (m, label)) in cands.enumerated() {
        guard m[0][0].isFinite, isSane(m, w: Double(flt.w), h: Double(flt.h)) else { continue }
        let s = score(ref: ref, flt: flt, h: m, stride: 4)
        if s.overlap >= 0.08, s.ncc > r.score.ncc {
            r = PairResult(h: m, score: s, method: label, convention: k % 8, ok: s.ncc >= minNCC)
        }
    }
    return r
}

// MARK: - Metal blend

let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Frame {
    float3x3 inv;      // canvas pixel -> source pixel
    float4 params;     // gain, width, height, unused
};

struct Globals {
    uint count;
    float power;
    int solo;
};

// Distance-weighted (feather) blend: every frame that covers an output pixel contributes with a
// weight that falls to zero at its own border, so seams fade instead of stepping.
kernel void warpBlend(texture2d_array<float, access::sample> src [[texture(0)]],
                      texture2d<float, access::write> dst [[texture(1)]],
                      constant Frame *frames [[buffer(0)]],
                      constant Globals &g [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::pixel, filter::linear, address::clamp_to_edge);
    float3 p = float3(float2(gid) + 0.5, 1.0);
    float3 acc = float3(0.0);
    float wsum = 0.0;
    for (uint i = 0; i < g.count; i++) {
        if (g.solo >= 0 && uint(g.solo) != i) continue;
        float3 q = frames[i].inv * p;
        if (q.z <= 1e-6) continue;
        float2 uv = q.xy / q.z;
        float w = frames[i].params.y, h = frames[i].params.z;
        if (uv.x < 0.5 || uv.y < 0.5 || uv.x > w - 0.5 || uv.y > h - 0.5) continue;
        float d = 2.0 * min(min(uv.x, w - uv.x) / w, min(uv.y, h - uv.y) / h);
        float wt = pow(d, g.power) + 1e-6;
        acc += src.sample(s, uv, i).rgb * frames[i].params.x * wt;
        wsum += wt;
    }
    dst.write(wsum > 0.0 ? float4(acc / wsum, 1.0) : float4(0.0, 0.0, 0.0, 1.0), gid);
}
"""

struct FrameUniform {
    var inv: simd_float3x3
    var params: SIMD4<Float>
}

struct GlobalsUniform {
    var count: UInt32
    var power: Float
    var solo: Int32
}

// MARK: - Main

let opt = parseOptions()
try FileManager.default.createDirectory(at: opt.out, withIntermediateDirectories: true)

guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
    fatalError("no Metal device")
}
let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
let ci = CIContext(mtlDevice: device, options: [
    .workingColorSpace: linear, .workingFormat: CIFormat.RGBAh, .cacheIntermediates: false,
])

var report: [String: Any] = [:]
report["machine"] = ["model": sysctlString("hw.model"), "chip": sysctlString("machdep.cpu.brand_string"),
                     "ramGB": Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824,
                     "os": ProcessInfo.processInfo.operatingSystemVersionString, "gpu": device.name]
report["options"] = ["scale": opt.scale, "regEdge": opt.regEdge, "power": opt.power, "gain": opt.gain, "lockWB": opt.lockWB]
let n = opt.inputs.count
let tStart = now()

// 1. Decode every RAW straight into one slice of a float16 texture array (linear sRGB, row 0 = top).
var frames: MTLTexture!
var slices: [MTLTexture] = []
var fw = 0, fh = 0
var decodeTimes: [Double] = []
var wb: (temp: Float, tint: Float)?
for (i, url) in opt.inputs.enumerated() {
    let t0 = now()
    guard let raw = CIRAWFilter(imageURL: url) else { fatalError("cannot open \(url.path)") }
    raw.scaleFactor = Float(opt.scale)
    if opt.lockWB {
        if let wb { raw.neutralTemperature = wb.temp; raw.neutralTint = wb.tint }
        else { wb = (raw.neutralTemperature, raw.neutralTint) }
    }
    guard var img = raw.outputImage else { fatalError("no image for \(url.path)") }
    img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
    let w = Int(img.extent.width.rounded()), h = Int(img.extent.height.rounded())
    if frames == nil {
        fw = w; fh = h
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .rgba16Float
        d.width = w; d.height = h; d.arrayLength = n
        d.usage = [.shaderRead, .shaderWrite, .renderTarget, .pixelFormatView]
        d.storageMode = .private
        frames = device.makeTexture(descriptor: d)!
    }
    guard w == fw, h == fh else { fatalError("frame sizes differ: \(url.lastPathComponent) is \(w)x\(h), expected \(fw)x\(fh)") }
    let slice = frames.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: 0..<1, slices: i..<i + 1)!
    let dest = CIRenderDestination(mtlTexture: slice, commandBuffer: nil)
    dest.colorSpace = linear
    dest.isFlipped = true
    _ = try ci.startTask(toRender: img, to: dest).waitUntilCompleted()
    slices.append(slice)
    decodeTimes.append(now() - t0)
    print(String(format: "decode  %@  %dx%d  %.3fs", url.lastPathComponent, w, h, decodeTimes[i]))
}
let tDecode = decodeTimes.reduce(0, +)

/// Upright CIImage over a texture whose row 0 is the top of the picture.
func uprightImage(_ tex: MTLTexture) -> CIImage {
    CIImage(mtlTexture: tex, options: [.colorSpace: linear])!
        .transformed(by: CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -CGFloat(tex.height)))
}

// 2. Register every pair with Vision at a reduced size, keep the pairs that really overlap,
//    and link the frames with a maximum spanning tree so shooting order does not matter.
let tRegPrep0 = now()
let regScale = min(1.0, Double(opt.regEdge) / Double(max(fw, fh)))
var regCG: [CGImage] = []
var grays: [Gray] = []
for s in slices {
    let img = uprightImage(s).transformed(by: CGAffineTransform(scaleX: regScale, y: regScale))
    let rect = CGRect(x: 0, y: 0, width: (Double(fw) * regScale).rounded(.down), height: (Double(fh) * regScale).rounded(.down))
    let cg = ci.createCGImage(img, from: rect, format: .RGBA8, colorSpace: srgb)!
    regCG.append(cg)
    grays.append(makeGray(cg))
}
let smalls = grays.map { shrink($0, width: 128) }
let tRegPrep = now() - tRegPrep0

let minNCC = 0.3
var pair: [[PairResult?]] = Array(repeating: Array(repeating: nil, count: n), count: n)
var pairRows: [[String: Any]] = []
let tReg0 = now()
for i in 0..<n {
    for j in (i + 1)..<n {
        let t0 = now()
        let r = register(refCG: regCG[i], fltCG: regCG[j], ref: grays[i], flt: grays[j], refSmall: smalls[i], fltSmall: smalls[j], minNCC: minNCC)
        pair[i][j] = r
        let dt = now() - t0
        print(String(format: "register %d<-%d  %@  ncc %.3f  overlap %.0f%%  mad %.3f  %@  %.3fs", i, j,
                     r.method as NSString, r.score.ncc, r.score.overlap * 100, r.score.mad, r.ok ? "ok" : "rejected", dt))
        pairRows.append(["ref": opt.inputs[i].lastPathComponent, "floating": opt.inputs[j].lastPathComponent,
                         "method": r.method, "convention": r.convention, "ncc": r.score.ncc,
                         "overlap": r.score.overlap, "mad": r.score.mad, "accepted": r.ok, "seconds": dt])
    }
}
let tReg = now() - tReg0
report["pairs"] = pairRows

// Kruskal, best-scoring pairs first.
var parent = Array(0..<n)
func find(_ x: Int) -> Int { var x = x; while parent[x] != x { x = parent[x] }; return x }
var edges: [(Int, Int)] = []
let ranked = (0..<n).flatMap { i in ((i + 1)..<n).map { (i, $0) } }
    .filter { pair[$0.0][$0.1]!.ok }
    .sorted { pair[$0.0][$0.1]!.score.ncc * pair[$0.0][$0.1]!.score.overlap.squareRoot()
            > pair[$1.0][$1.1]!.score.ncc * pair[$1.0][$1.1]!.score.overlap.squareRoot() }
for (i, j) in ranked where find(i) != find(j) {
    parent[find(i)] = find(j)
    edges.append((i, j))
}
guard edges.count == n - 1 else {
    print("FAILED: frames do not form one connected panorama (\(edges.count) usable links for \(n) frames).")
    report["failed"] = "not connected"
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: opt.out.appendingPathComponent("\(opt.name)-metrics.json"))
    exit(1)
}

// Chain homographies (and exposure gains) from each candidate reference; keep the smallest canvas.
let up = Double(fw) / Double(grays[0].w)
let toWork = m3([up, 0, 0], [0, up, 0], [0, 0, 1])
func link(_ a: Int, _ b: Int) -> (M3, Double) {
    // Transform and linear gain taking frame b into frame a.
    if a < b { let r = pair[a][b]!; return (r.h, r.score.linRatio) }
    let r = pair[b][a]!
    return (r.h.inverse, 1 / r.score.linRatio)
}
func chain(root: Int) -> ([M3], [Double]) {
    var hs = [M3](repeating: matrix_identity_double3x3, count: n)
    var gs = [Double](repeating: 1, count: n)
    var seen: Set<Int> = [root]
    var todo = [root]
    while let a = todo.popLast() {
        for (i, j) in edges where i == a || j == a {
            let b = i == a ? j : i
            guard !seen.contains(b) else { continue }
            let (h, g) = link(a, b)
            hs[b] = normalized(hs[a] * h)
            gs[b] = gs[a] * g
            seen.insert(b); todo.append(b)
        }
    }
    return (hs.map { normalized(toWork * $0 * toWork.inverse) }, gs)
}
func bounds(_ hs: [M3]) -> CGRect {
    var r = CGRect.null
    for h in hs {
        for (x, y) in [(0.0, 0.0), (Double(fw), 0.0), (Double(fw), Double(fh)), (0.0, Double(fh))] {
            let p = apply(h, x, y)
            r = r.union(CGRect(x: p.x, y: p.y, width: 0, height: 0))
        }
    }
    return r
}
var root = 0
var hs: [M3] = []
var gains: [Double] = []
var box = CGRect.null
for r in 0..<n {
    let (h, g) = chain(root: r)
    let b = bounds(h)
    if box.isNull || b.width * b.height < box.width * box.height { root = r; hs = h; gains = g; box = b }
}
if opt.gain {
    // Centre the gains so the panorama as a whole keeps its exposure.
    let mean = exp(gains.map { log($0) }.reduce(0, +) / Double(n))
    gains = gains.map { $0 / mean }
} else {
    gains = [Double](repeating: 1, count: n)
}

let maxDim = 16000.0
let fit = min(1.0, maxDim / max(box.width, box.height))
let cw = Int((box.width * fit).rounded(.up)), ch = Int((box.height * fit).rounded(.up))
let toCanvas = m3([fit, 0, -box.minX * fit], [0, fit, -box.minY * fit], [0, 0, 1])
let full = hs.map { normalized(toCanvas * $0) }
print("reference frame: \(opt.inputs[root].lastPathComponent)   canvas \(cw)x\(ch)   gains \(gains.map { String(format: "%.3f", $0) })")

// 3. Warp + blend in one compute pass.
let tSetup0 = now()
let library = try device.makeLibrary(source: shaderSource, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "warpBlend")!)
let tSetup = now() - tSetup0

let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: cw, height: ch, mipmapped: false)
cd.usage = [.shaderRead, .shaderWrite]
cd.storageMode = .private
let canvas = device.makeTexture(descriptor: cd)!

var uniforms: [FrameUniform] = (0..<n).map { i in
    let inv = normalized(full[i].inverse)
    let c = inv.columns
    return FrameUniform(inv: simd_float3x3(SIMD3<Float>(c.0), SIMD3<Float>(c.1), SIMD3<Float>(c.2)),
                        params: SIMD4(Float(gains[i]), Float(fw), Float(fh), 0))
}
var globals = GlobalsUniform(count: UInt32(n), power: opt.power, solo: Int32(opt.solo))

let tBlend0 = now()
let cb = queue.makeCommandBuffer()!
let enc = cb.makeComputeCommandEncoder()!
enc.setComputePipelineState(pipeline)
enc.setTexture(frames, index: 0)
enc.setTexture(canvas, index: 1)
enc.setBytes(&uniforms, length: MemoryLayout<FrameUniform>.stride * n, index: 0)
enc.setBytes(&globals, length: MemoryLayout<GlobalsUniform>.stride, index: 1)
let tw = pipeline.threadExecutionWidth, th = pipeline.maxTotalThreadsPerThreadgroup / pipeline.threadExecutionWidth
enc.dispatchThreadgroups(MTLSize(width: (cw + tw - 1) / tw, height: (ch + th - 1) / th, depth: 1),
                         threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
enc.endEncoding()
cb.commit()
cb.waitUntilCompleted()
if let e = cb.error { fatalError("blend failed: \(e)") }
let tBlend = now() - tBlend0
let tBlendGPU = cb.gpuEndTime - cb.gpuStartTime
let tPipeline = now() - tStart
print(String(format: "warp+blend %.3fs wall (%.3fs GPU), shader compile %.3fs", tBlend, tBlendGPU, tSetup))

// 4. Outputs: full JPEG, ~2000 px preview, 100 % crops centred on the seams.
let tEnc0 = now()
let pano = uprightImage(canvas)
let jpegOpts: [CIImageRepresentationOption: Any] = [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9]
let fullURL = opt.out.appendingPathComponent("\(opt.name)-full.jpg")
try ci.writeJPEGRepresentation(of: pano, to: fullURL, colorSpace: srgb, options: jpegOpts)
let tEncFull = now() - tEnc0

let ps = min(1.0, 2000.0 / Double(cw))
let lanczos = CIFilter(name: "CILanczosScaleTransform", parameters: [kCIInputImageKey: pano, kCIInputScaleKey: ps, kCIInputAspectRatioKey: 1.0])!
let preview = lanczos.outputImage!.cropped(to: CGRect(x: 0, y: 0, width: (Double(cw) * ps).rounded(.down), height: (Double(ch) * ps).rounded(.down)))
try ci.writeJPEGRepresentation(of: preview, to: opt.out.appendingPathComponent("\(opt.name)-preview.jpg"), colorSpace: srgb, options: jpegOpts)

// A seam sits where two linked frames carry equal weight: half way between their warped centres.
var cropRows: [[String: Any]] = []
for (k, e) in edges.prefix(opt.crops).enumerated() {
    let a = apply(full[e.0], Double(fw) / 2, Double(fh) / 2), b = apply(full[e.1], Double(fw) / 2, Double(fh) / 2)
    let w = min(opt.cropSize.w, cw), h = min(opt.cropSize.h, ch)
    let x = min(max(0, Int((a.x + b.x) / 2) - w / 2), cw - w)
    let y = min(max(0, Int((a.y + b.y) / 2) - h / 2), ch - h)
    let rect = CGRect(x: x, y: ch - y - h, width: w, height: h)   // Core Image is y-up
    let img = pano.cropped(to: rect).transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
    let name = "\(opt.name)-seam\(k + 1).jpg"
    try ci.writeJPEGRepresentation(of: img, to: opt.out.appendingPathComponent(name), colorSpace: srgb, options: jpegOpts)
    cropRows.append(["file": name, "frames": [opt.inputs[e.0].lastPathComponent, opt.inputs[e.1].lastPathComponent],
                     "x": x, "y": y, "w": w, "h": h])
}

let peak = peakFootprintBytes()
report["frames"] = opt.inputs.map(\.lastPathComponent)
report["workingSize"] = [fw, fh]
report["registrationSize"] = [grays[0].w, grays[0].h]
report["canvas"] = [cw, ch]
report["reference"] = opt.inputs[root].lastPathComponent
report["links"] = edges.map { [opt.inputs[$0.0].lastPathComponent, opt.inputs[$0.1].lastPathComponent] }
report["gains"] = gains
report["crops"] = cropRows
report["seconds"] = [
    "decodeTotal": tDecode, "decodePerFrame": decodeTimes,
    "registrationPrep": tRegPrep, "registrationTotal": tReg,
    "registrationVisionCalls": visionSeconds, "registrationCoarseSearch": coarseSeconds,
    "shaderCompile": tSetup, "warpBlendWall": tBlend, "warpBlendGPU": tBlendGPU,
    "pipelineTotal": tPipeline, "encodeFullJPEG": tEncFull,
]
report["peakFootprintMB"] = Double(peak) / 1_048_576
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    .write(to: opt.out.appendingPathComponent("\(opt.name)-metrics.json"))

print(String(format: "vision calls %.2fs, coarse search %.2fs, scoring %.2fs", visionSeconds, coarseSeconds, tReg - visionSeconds - coarseSeconds))
print(String(format: "decode %.2fs | registration %.2fs (+%.2fs prep) | warp+blend %.3fs | pipeline %.2fs | full JPEG %.2fs | peak footprint %.0f MB",
             tDecode, tReg, tRegPrep, tBlend, tPipeline, tEncFull, Double(peak) / 1_048_576))
