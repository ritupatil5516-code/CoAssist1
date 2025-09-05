from pathlib import Path
import yaml

class PromptPack:
    def __init__(self, root: Path):
        self.root = root
        self.system = (root / "system.md").read_text(encoding="utf-8")
        self.answer_style = (root / "answer_style.md").read_text(encoding="utf-8")
        self.refusal = (root / "refusal.md").read_text(encoding="utf-8")
        self.retrieval = (root / "retrieval.md").read_text(encoding="utf-8")
        self.glossary = (root / "glossary.md").read_text(encoding="utf-8")
        self.instructions = yaml.safe_load((root / "instructions.yaml").read_text(encoding="utf-8"))

def load_prompts(prompts_dir):
    return PromptPack(Path(prompts_dir))
