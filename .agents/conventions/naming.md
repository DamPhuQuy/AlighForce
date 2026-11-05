# Naming and Code Conventions

<naming_conventions version="1.0">

Follow the language, framework, and repository's existing conventions. Prefer names that reveal purpose at the point of use. Expand an abbreviation when it obscures meaning; common domain terms and language idioms are fine.

Use a leading underscore when the language or codebase uses it for a real convention, such as Python nonpublic members. Do not rename unrelated identifiers to satisfy this file.

Keep a function or class cohesive, but split it only when doing so improves comprehension or testing. Extract a literal into a named constant when its meaning or reuse warrants it; small obvious values need no ceremony.

Address relevant lint and type errors in changed code. A narrow, reviewed suppression with a reason may be appropriate when a tool cannot express a valid construct. Do not silence a real failure merely to pass a check.

</naming_conventions>
