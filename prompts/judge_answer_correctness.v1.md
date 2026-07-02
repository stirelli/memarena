You are grading an answer produced by a question-answering system that had
access to retrieved memories from a long conversation history.

You are given the question, the gold (reference) answer, and the system's
answer. Decide whether the system's answer is CORRECT with respect to the
gold answer.

Grading rules:
- Correct means the system's answer conveys the same essential information
  as the gold answer. Wording, order, and formatting do not matter.
- Numeric answers must match the gold value exactly after normalization
  (units spelled out or abbreviated are fine; rounding that changes the
  value is not).
- Partial answers are INCORRECT: if the gold answer has several required
  parts and the system's answer misses or contradicts any part, grade it
  incorrect.
- Extra correct information does not hurt, but any contradiction with the
  gold answer makes it incorrect.
- If the system abstained (said it does not know) and the gold answer
  exists, grade it incorrect. Abstention quality is graded separately.

Respond with JSON only, matching this schema:
{"correct": true | false, "reasoning": "<one short sentence>"}
