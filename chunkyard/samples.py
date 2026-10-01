"""Built-in sample documents and their suggested questions."""

from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"

# Built-in samples. A sample whose file is missing shows as unavailable.
SAMPLES = {
    "tent": {
        "file": "trailhead-tent-manual.md",
        "title": "Tent manual",
        "blurb": "Short and tidy. The strategies often agree.",
        "questions": [
            "How long is the warranty?",
            "What isn't covered?",
            "How should I store the tent over the winter?",
            "Why is the inside of my tent wet in the morning?",
            "How much does it weigh?",
        ],
    },
    "fda": {
        "file": "repatha-fda-label.pdf",
        "title": "FDA drug label (PDF)",
        "blurb": "65 pages of numbered sections, bullet lists, and results tables. Expect disagreement.",
        "questions": [
            "What is the recommended dose for adults with HoFH?",
            "What is Repatha used to treat?",
            "How much did 420 mg once monthly lower LDL-C compared with placebo?",
            "How should Repatha be stored?",
            "What should a patient do if they miss a dose?",
        ],
    },
}
