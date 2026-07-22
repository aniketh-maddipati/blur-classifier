from pathlib import Path

from PIL import Image

from contact_sheet import SheetEntry, default_output_path, render_contact_sheet


def test_render_contact_sheet_produces_nonempty_image(tmp_path: Path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (200, 100), (255, 0, 0)).save(image_a)
    Image.new("RGB", (100, 200), (0, 255, 0)).save(image_b)

    sheet = render_contact_sheet(
        [
            SheetEntry("DSC0001", image_a, "sharp", "intentional_blur"),
            SheetEntry("DSC0002", image_b, "unintentional_blur", "sharp"),
        ],
        single_row=False,
    )

    assert sheet.width > 0
    assert sheet.height > 0


def test_default_output_path_uses_results_directory():
    assert default_output_path(Path("results/eval_runC_1344.csv"), single_row=False) == Path(
        "results/eval_runC_1344_misses.png"
    )
