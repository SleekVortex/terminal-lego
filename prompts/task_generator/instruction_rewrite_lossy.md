Rewrite the task below as a brief, deliberately under-specified request that a real developer or user might send.

Write naturally, preferably starting with a concrete symptom such as "I have...", "Something is wrong...", or "I need...".

Keep only:
- the core problem or symptom;
- the high-level desired outcome;
- truly non-negotiable constraints that cannot be discovered by inspecting the provided files;
- at most the top-level working location and one essential final destination or artifact, when needed.

Intentionally leave out details that a capable solver can discover from the environment. Deliberately omit:
- the root cause and expected diagnosis;
- recommended fixes, implementation approaches, algorithms, and code snippets;
- commands and step-by-step instructions;
- exhaustive file lists and intermediate artifact names;
- detailed output schemas unless the schema is itself the requested public interface;
- acceptance-test checklists, verification commands, expected diagnostic values, hints, and background explanations.

Do not preserve every detail. Do not turn the request into a checklist. Do not mention StackOverflow, a verifier, a reference solution, hidden tests, task generation, or container internals.

Do not change the core problem, contradict the provided environment, invent requirements, or remove so much that the requested final outcome cannot be identified at all. The request should force the solver to inspect files, reproduce the issue, and determine the implementation independently.

Target 35-90 words. Use up to 120 words only when an essential public interface or data format cannot be described more briefly.

Task to rewrite:
{instruction}

Output only the rewritten request.
