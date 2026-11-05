# Testing Context

<testing_context version="2.0">

## Choose the smallest useful verifier

<test_pyramid>
Use a unit test for local deterministic logic, an integration test when components or persistence interact, and an end to end check for a critical journey or public boundary. Follow the repository's test layout and commands; the labels do not prescribe directories, execution times, or a fixed number of tests. Run broader checks when the changed contract can affect distant callers.
</test_pyramid>

<mocking_boundaries>
Mock external services, time, or other expensive and unstable boundaries when that makes the test reliable. Prefer assertions on observable output and state over private call counts. Do not mock the behavior under test. A real boundary check is useful when mocks cannot establish the contract.
</mocking_boundaries>

<fixture_hygiene>
Keep test state isolated and clean it up. Use repository fixtures or factories when they make intent clear. A shared fixture is acceptable when it is immutable or reset reliably; investigate order dependent or flaky tests before calling a new failure a regression.
</fixture_hygiene>

## Defects and changes

<adaptive_tdd_policy>
For a defect, establish an observable failure before editing when feasible. A focused regression test is valuable when it will catch the same failure later. If reproduction is intermittent or unavailable, record the limitation and use logs, traces, or another direct observation. For a new contract, tests may precede or follow implementation according to the risk and repository workflow. For documentation, assets, or mechanical changes, a focused lint, schema, visual, or diff check may be sufficient. Do not add a test that merely repeats implementation details.
</adaptive_tdd_policy>

When an existing test encodes a demonstrably wrong expectation, update it and explain the changed contract. Never weaken an assertion solely to make the suite pass. Report the command and outcome for each material check, including unavailable or pre-existing failures.

</testing_context>
