# Putting this on GitHub

```bash
cd ape-classification-tool
git init
git add .
git commit -m "Initial commit: APE classification tool (Algorithm 1, baselines, GUIs, definitions)"
git branch -M main
git remote add origin https://github.com/<you>/ape-classification-tool.git
git push -u origin main
```

The included CI workflow (`.github/workflows/tests.yml`) runs the test suite on
every push and pull request.

## Suggested repo settings
- Add topics: `prompt-engineering`, `llm`, `text-classification`,
  `requirements-engineering`, `ollama`, `gradio`.
- Enable GitHub Pages from `figures/` if you want to preview the PDFs.
- Replace the citation block and `<REPLACE-WITH-REPOSITORY-URL>` in
  `docs/tool_subsection.tex` with the final DOI / URL.
