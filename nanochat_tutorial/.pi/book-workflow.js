const reviews = await runs.all([
  {
    key: "review-foundations",
    label: "Review foundations chapters",
    agent: "reviewer",
    task: [
      "Goal: review tutorial chapters 01-05 for publication-quality improvements inspired by strong codebase documentation practices, without mentioning or citing DeepWiki anywhere in the book.",
      "Target: cwd repository, fixed nanochat source revision already present. Read tutorial/README.md, tutorial/01-overview.md through tutorial/05-gpt-mlp-tricks.md, and relevant nanochat source/tests as needed.",
      "Authority: read-only. Do not edit project files.",
      "Assess chapter title/style consistency, opening Purpose/Scope/Input/Output/source map, lifecycle/entity mapping, shape continuity, redundancy, transitions, code accuracy, and what a printed technical book needs.",
      "Return a concise actionable review grouped by file, with exact headings/locations, smallest useful edits, and any facts that must be rechecked. Do not recommend adding references to the external inspiration source.",
      "Stop after identifying the highest-value concrete improvements; avoid broad rewrites without evidence."
    ].join("\n")
  },
  {
    key: "review-training",
    label: "Review training chapters",
    agent: "reviewer",
    task: [
      "Goal: review tutorial chapters 06-08 for publication-quality improvements, without mentioning or citing DeepWiki anywhere in the book.",
      "Target: tutorial/06-base-train-loop.md, tutorial/07-base-train-optim-scaling.md, tutorial/08-base-evaluation.md, plus relevant nanochat scripts/source/tests.",
      "Authority: read-only. Do not edit project files.",
      "Focus on script lifecycle maps, complexity-dial derivation, optimizer parameter map, config defaults versus speedrun overrides, evaluation/checkpoint boundaries, performance cost accounting, section balance, and factual accuracy at revision 92d63d4.",
      "Return evidence-backed, file-specific edits and flag duplication or overlong sections. Do not name the external inspiration source."
    ].join("\n")
  },
  {
    key: "review-chat",
    label: "Review chat and RL chapters",
    agent: "reviewer",
    task: [
      "Goal: review tutorial chapters 09-13 for publication-quality improvements, without mentioning or citing DeepWiki anywhere in the book.",
      "Target: tutorial/09-sft.md through tutorial/13-end-to-end.md and relevant nanochat source/tests.",
      "Authority: read-only. Do not edit project files.",
      "Focus on SFT/inference/chat-eval/RL lifecycle continuity; concept-to-code entity maps; task/verifier boundaries; checkpoint and CLI accuracy; safety caveats; chapter inputs/outputs; cross-links; and end-to-end closure.",
      "Return concise file-specific findings with exact sections and smallest fixes. Do not name the external inspiration source."
    ].join("\n")
  },
  {
    key: "review-apparatus",
    label: "Review appendices and book apparatus",
    agent: "reviewer",
    task: [
      "Goal: review all tutorial appendices, README references, glossary, checks, and publication apparatus for a coherent technical book, without mentioning or citing DeepWiki anywhere in the book.",
      "Target: tutorial/README.md, glossary.md, toy-checks.md, performance-research.md, source_checks.py, deepwiki-audit.md, and links from chapters.",
      "Authority: read-only. Do not edit files.",
      "Design a reader-facing source/file index and testing/validation guide; decide how to replace/remove the external-source audit; check appendix lettering/order, terminology/code pointers, local links, source check assumptions, and publication tone.",
      "Return a concrete proposed appendix structure and file-level actions. No external inspiration source may remain in filenames or prose."
    ].join("\n")
  }
]);

const writer = await runs.run("book-writer", {
  label: "Rewrite book chapters",
  agent: "worker",
  task: [
    "Goal: perform a complete publication-quality optimization of the nanochat tutorial book, modifying tutorial files only.",
    "The user explicitly requires that the external inspiration source never appear in the book: remove every case-insensitive mention from tutorial prose, links, headings, metadata, and filenames. Rename or replace tutorial/deepwiki-audit.md with reader-facing material and update every link. README was already rewritten; preserve its strong book-like structure while removing forbidden mentions and integrating the final appendix organization.",
    "Use the four independent reviews below as advisory evidence. Inspect actual files/source yourself and apply only correct, current-revision changes.",
    "Required scope across every chapter except README (README still needs reference cleanup and link integration):",
    "- Standardize chapter openings so each clearly states position, central question, scope/non-scope, inputs/outputs, prerequisites, and source entry points without bloating prose.",
    "- Add compact concept-to-code entity maps and script lifecycle summaries where they materially aid navigation; reuse existing material and avoid duplicate diagrams.",
    "- Improve part-to-part transitions, terminology consistency, captions/labels for important tables or figures where useful, and publication tone.",
    "- Preserve technical depth, formulas, exercises, code examples, and current-source corrections. Do not blindly rewrite correct chapters or inflate length.",
    "- Create a reader-facing source topic/file index appendix and a testing/validation guide appendix if that is the cleanest design. Update glossary to include code entity pointers if practical.",
    "- Ensure appendices have coherent labels/order and all local links resolve.",
    "- Update tutorial/source_checks.py so it enforces: no forbidden case-insensitive term anywhere under tutorial (except source_checks.py may contain an obfuscated construction rather than the literal), all local links valid, existing source contracts still checked. The literal forbidden word must not appear anywhere under tutorial, including code and filenames.",
    "- Rename files with shell mv when needed; do not touch nanochat source, git history, or files outside tutorial.",
    "Validation required: python3 tutorial/toy_checks.py; python3 tutorial/source_checks.py; python3 -m py_compile tutorial/*.py; git diff --check -- tutorial; case-insensitive recursive grep for the forbidden term must return no matches; inspect final diff/stat.",
    "Report changed/renamed files, structural improvements, validation results, and residual risks. Stop and ask only for a material editorial decision that cannot be inferred.",
    "\nFOUNDATIONS REVIEW:\n" + reviews[0].output,
    "\nTRAINING REVIEW:\n" + reviews[1].output,
    "\nCHAT/RL REVIEW:\n" + reviews[2].output,
    "\nAPPARATUS REVIEW:\n" + reviews[3].output
  ].join("\n")
});

const validations = await runs.all([
  {
    key: "validate-publication",
    label: "Validate publication quality",
    agent: "reviewer",
    task: [
      "Review the post-writer tutorial as a fresh technical-book editor. Read the actual tutorial files and current diff.",
      "Do not edit files. Check all chapters and appendices for consistent openings, coherent hierarchy, repetition, transitions, terminology, reader navigation, appendix order, and publication tone. Verify the forbidden external-source name is absent without writing it in your response; refer to it only as 'the forbidden term'.",
      "Return only evidence-backed P0/P1/P2 findings with file/heading references and smallest safe fixes. End with Merge verdict.",
      "Writer report: " + writer.output
    ].join("\n")
  },
  {
    key: "validate-technical",
    label: "Validate technical accuracy",
    agent: "reviewer",
    task: [
      "Review the post-writer tutorial against nanochat revision 92d63d4. Read actual source/tests and diff. Do not edit files.",
      "Check high-risk facts: pipeline stages, tokenizer/data order, model architecture/current constants, initialization, optimizer grouping/schedules/distribution, approximate resume, SFT mixture, five-task ChatCORE, inference/KV state, RL simplifications, CLI/checkpoint paths, FP8 limitations, and benchmark boundaries.",
      "Also inspect source_checks.py and local links. Report only concrete P0/P1/P2 findings with file/heading evidence and smallest fixes. End with Merge verdict.",
      "Writer report: " + writer.output
    ].join("\n")
  }
]);

const fixer = await runs.run("book-fixer", {
  label: "Fix reviewed book issues",
  agent: "worker",
  task: [
    "Goal: apply only valid P0/P1 and straightforward P2 findings from the two fresh reviews to tutorial files. You are the sole final writer.",
    "Inspect actual current files and diff. Reject findings contradicted by source or user scope. Do not mention or cite the forbidden external inspiration source anywhere, and ensure no literal occurrence remains in tutorial paths or contents.",
    "Preserve publication-quality structure and avoid scope expansion. Run all validations: toy checks, source checks, py_compile, diff check, local link checks through source_checks, and case-insensitive grep for the forbidden term. Inspect final diff and report disposition of each finding plus residual risks.",
    "\nPUBLICATION REVIEW:\n" + validations[0].output,
    "\nTECHNICAL REVIEW:\n" + validations[1].output
  ].join("\n")
});

return {
  reviews: reviews.map((result) => result.output),
  writer: writer.output,
  validations: validations.map((result) => result.output),
  fixer: fixer.output
};
