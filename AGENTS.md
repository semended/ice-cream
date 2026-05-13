# Project Instructions for Codex

## Project context

This project is an MVP for VLM-based analysis of ice-cream freezer photos.

The business goal is to analyze photos of retail freezer equipment and evaluate the representation of the brand "Коровка из Кореновки" / KIK.

The system should return structured JSON with fields such as:

- whether KIK products are visible;
- visible KIK SKU count;
- KIK share in the freezer;
- present KIK SKU families;
- POSM presence;
- monobrand block presence;
- whether KIK is mixed with competitors;
- foreign products;
- empty sections;
- final outlet status;
- confidence.

## Core rule

Do not just write code.

Before changing code, always understand the business problem, the affected JSON fields, the scoring logic, and the expected evaluation effect.

Optimize for measurable improvement on the labeled photo dataset, not for beautiful abstractions.

Prefer simple, robust rules over complex logic when they solve the business problem.

Do not invent facts about the data. If data is missing, explicitly say what must be inspected or labeled.

## Required problem-solving workflow

For every task, follow this workflow:

1. Restate the business problem in simple terms.
2. Identify which JSON fields and scoring components are affected.
3. Classify the model/evaluation issue using the error taxonomy below.
4. Generate root-cause hypotheses.
5. Propose fix options across:
   - prompt;
   - JSON schema;
   - scoring logic;
   - dataset / labels;
   - photo guidelines;
   - model / provider choice;
   - evaluation pipeline.
6. Prioritize fixes by Impact / Effort / Risk.
7. Only then propose code changes.
8. For every code change, explain:
   - files to edit;
   - what exactly changes;
   - why it changes;
   - how to test it;
   - expected effect on evaluation.
9. If a decision changes the project logic, update or propose an entry for `docs/decision_log.md`.

## Error taxonomy

Use these error types:

1. False positive:
   Model sees KIK when KIK is not actually visible.

2. False negative:
   Model misses KIK when KIK is visible.

3. SKU count error:
   Model gives wrong number of visible KIK SKUs.

4. SKU-family error:
   Model misclassifies product families, for example confusing bricket and poleno.

5. KIK share error:
   Model estimates the wrong KIK share in the freezer.

6. Perspective/photo-angle issue:
   Error caused by bad shooting angle, perspective distortion, or distant products.

7. Small-object visibility issue:
   Error caused by small or rare KIK products, for example one bucket in the freezer.

8. Business-status interpretation error:
   Model detects visual facts correctly but assigns the wrong outlet status.

9. JSON/schema error:
   Output is invalid JSON or does not match the expected schema.

10. Hallucination:
   Model invents a product, SKU, POSM, or fact that is not visible.

## Fix types

When solving a problem, consider these fix types:

### Prompt fix

Use when the model misunderstands the task or business meaning.

Examples:
- clarify that KIK share means KIK / (KIK + competitors), not KIK / whole freezer volume;
- forbid inventing SKU families;
- explain how to treat bricket and poleno.

### Schema fix

Use when the current JSON structure forces bad decisions.

Examples:
- merge bricket and poleno into one field;
- split visual facts from business conclusions;
- add `unknown` instead of forcing yes/no.

### Scoring fix

Use when the evaluation punishes or rewards the wrong thing.

Examples:
- relative error should matter more when ground truth is small;
- status should have high weight;
- all SKU-family weights should be equal unless there is a business reason.

### Dataset / label fix

Use when labels are inconsistent or missing.

Examples:
- label whether photo angle is acceptable;
- label visible KIK groups separately from exact SKU count;
- add difficult examples with one small KIK product.

### Photo-guideline fix

Use when the model error is caused by photo quality or shooting angle.

Examples:
- require photo from above;
- avoid steep side angles;
- capture the whole freezer.

### Model/provider fix

Use when the current model cannot reliably solve the visual task.

Examples:
- compare lightweight local VLMs against heavy benchmark models;
- test structured output support;
- check JSON reliability.

### Evaluation pipeline fix

Use when the benchmark does not measure business success correctly.

Examples:
- separate field-level score from final business status score;
- log per-field errors;
- create error reports by photo and by model.

## Coding principles

- Keep code simple and readable.
- Avoid large rewrites unless necessary.
- Do not silently change business logic.
- Add tests or evaluation checks for scoring changes.
- Keep prompts versioned.
- Keep scoring logic explainable.
- Separate raw model output, parsed JSON, scoring, and reporting.
- Never hide failed parsing or invalid JSON.
- Log enough information to debug model mistakes.

## Expected output from Codex

When responding to a task, use this format:

1. Business problem
2. Affected fields / metrics
3. Error type
4. Hypotheses
5. Fix options
6. Recommended fix
7. Files to change
8. Test plan
9. Expected evaluation effect
10. Code changes
