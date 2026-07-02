You are auditing the RETRIEVAL stage of a memory system. The system was
asked a question about a long conversation history; you are given the
question, the gold answer, the gold evidence (the original conversation
turns that contain the needed information), and the set of memories the
system retrieved.

Decide whether the information needed to produce the gold answer is
PRESENT in the retrieved set.

Grading rules:
- Paraphrased content counts. A retrieved memory that restates the needed
  facts in different words covers the evidence. You are scoring
  information presence, not string overlap.
- The needed information may be spread across several retrieved memories;
  coverage is judged over the whole set.
- Coverage requires everything needed for the gold answer: if the gold
  answer needs two facts (for example two trip durations) and only one is
  present, coverage is false.
- Ignore answer correctness entirely: do not judge what the system
  answered, only what it retrieved.
- Related-but-insufficient content (the right topic without the needed
  facts) is NOT coverage.

Respond with JSON only, matching this schema:
{"covered": true | false, "reasoning": "<one short sentence>"}
