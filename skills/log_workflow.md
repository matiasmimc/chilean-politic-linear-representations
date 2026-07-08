# Skill: Project Log & Collaborative Workflow Maintenance

## Objective
This guideline ensures that the project's workflow, experiments, and results are consistently documented in `log.md`. By following these instructions, any collaborator or AI assistant will maintain a clear history of what has been tried, what succeeded, and what failed.

## Triggers
Follow this workflow whenever:
- A new experiment is designed or executed.
- New steering or evaluation results are generated (e.g., after running the evaluation script).
- Key parameters are found to break or improve the model (e.g., an `alpha` value that destroys coherence).
- Significant structural changes are made to the repository.

## Instructions
1. **Read Before Acting**: Before starting a new task or running an experiment, briefly read the "Experiment Log" and "Project Roadmap" in `log.md` to understand the current context and avoid repeating failed approaches.
2. **Document the Setup**: When you start a meaningful experiment or run the pipeline, add a new entry to the bottom of the "Experiment Log" in `log.md`. Include the Date, the Experiment Name, and the Objective.
3. **Document the Findings**: Once results are obtained (e.g., `intervention_evaluated.parquet` is generated or figures are analyzed), update the entry with:
   - **Status**: Success, Failed, or Needs Tweaking.
   - **Insights**: Note what worked (e.g., "alpha=20 shifts ideology effectively") and what didn't (e.g., "alpha > 30 causes severe repetition loops").
4. **Update the Roadmap**: If a major milestone is reached or a new blocker is discovered, check off or add items in the "Project Roadmap" section of `log.md`.
5. **Keep It Concise**: Write direct and to-the-point summaries. Use bullet points. Do not paste raw data dumps into the log.

## Example Entry Format
```markdown
### Date: YYYY-MM-DD
- **Experiment**: [Name of the experiment/run]
- **Objective**: [What we tried to find out or fix]
- **Status**: [Completed / In Progress / Failed]
- **Insights**: [Key takeaways, metrics, or bugs discovered]
```
