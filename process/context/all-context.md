# Root Context Router (SACH-2026 Framework)

<context_router version="4.0" framework="SACH">

<description>
  Central durable knowledge index routing to specialized domain context groups across the system.
</description>

## Context Directory Index

<!--
  CONTEXT ROUTING GUIDELINES:
  - Central index for all project domain knowledge, architecture, and invariants.
  - Register new context groups here when they become durable and reusable.
  - Specify clear `scope` so agents know exactly when to load this context group without
    unnecessary full-directory scanning.
-->
<context_groups>
  <group id="planning">
    <title>Planning Standards & Calibration</title>
    <path>[`planning/all-planning.md`](planning/all-planning.md)</path>
    <scope>Outcome, scope, coherent slices, and rollback reasoning when relevant</scope>
  </group>

  <group id="schemas">
    <title>Spec & Task Schemas</title>
    <path>[`../../.agents/schemas/spec.schema.json`](../../.agents/schemas/spec.schema.json)</path>
    <scope>EARS requirements syntax, BDD criteria, and Task DAG specification</scope>
  </group>

  <group id="tests">
    <title>Testing Standards & Harness Strategy</title>
    <path>[`tests/all-tests.md`](tests/all-tests.md)</path>
    <scope>Choose relevant checks, fixtures, and isolation boundaries</scope>
  </group>

  <group id="protocols">
    <title>Development Protocols Index</title>
    <path>[`../development-protocols/all-development-protocols.md`](../development-protocols/all-development-protocols.md)</path>
    <scope>Engineering harness, subagent orchestration, and code standards</scope>
  </group>

  <!-- Example: Register custom domain context below
  <group id="domain-billing">
    <title>Billing & Invoicing Architecture</title>
    <path>[`billing/billing-context.md`](billing/billing-context.md)</path>
    <scope>Payment gateway integrations, invoice calculation invariants, tax handling</scope>
  </group>
  -->
</context_groups>

</context_router>
