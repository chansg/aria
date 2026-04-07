# Git Workflow for Project Aria

## Every time you make changes:
```
git add .
git commit -m "describe what you changed"
git push
```

## Recommended commit message format:
```
feat: add voice training module
fix: correct sprite blink timing
refactor: reduce Claude API calls in brain.py
docs: update README with Coqui TTS setup
```

## Rules:
- Never commit `config.py`, `.env`, or anything in `data/`
- Verify `.gitignore` is working before every push
- If anything looks wrong with sensitive files — stop and flag it immediately
