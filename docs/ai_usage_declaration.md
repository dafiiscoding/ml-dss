# AI Usage Declaration

## Tools used

- An AI assistant in the original Antigravity development environment.
- Claude for planning, code generation, and project restructuring.
- OpenAI Codex for audit, refactoring, verification, dashboard QA, and report completion.

## Assisted work

AI assistance was used for:

- project scaffolding and implementation suggestions;
- regex/text preprocessing, CLIP integration, and model-training code;
- Streamlit and Plotly code;
- notebook and LaTeX structure;
- debugging, test creation, and consistency review.

## Human responsibility and verification

AI-generated text or code is not treated as evidence by itself. The project is
verified through:

1. Real CrisisMMD annotations and 18,082 referenced images.
2. Official informative labels joined by tweet/image identity rather than
   inferred from humanitarian categories.
3. SHA-256 and cleaned-text audit, with prior-split duplicates excluded from
   dev/test metrics.
4. Executed train-only EDA and modeling notebooks with embedded outputs.
5. Metrics generated from code using common official targets.
6. Thirty-seven automated tests and a real-image end-to-end integration test.
7. Streamlit AppTest for every page and browser screenshots.
8. Robust-mask sensitivity, stratified bootstrap, and event/class stability.
9. A XeLaTeX report compiled and visually rendered for inspection.

The team remains responsible for checking the analysis, supplying true member
information, and making all final academic claims.
