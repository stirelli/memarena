You are classifying whether a question-answering system ABSTAINED.

You are given the system's answer text. Decide whether it is an
abstention: a statement that it does not know, cannot answer, or lacks
the information, INSTEAD of an attempt to answer.

Rules:
- "I don't know", "I don't have that information", "the context does not
  mention it", and close variants are abstentions.
- An answer that hedges but still commits to a value ("I believe it was
  4 days") is NOT an abstention.
- An answer that gives a value and then disclaims confidence is NOT an
  abstention.

Respond with JSON only, matching this schema:
{"abstained": true | false, "reasoning": "<one short sentence>"}
