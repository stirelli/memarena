# Scoring rubric v1 (Day 4)

Versioned with judge.v1 (configs/judges/judge.v1.yaml): human labelers and
the LLM judges grade against THIS document. Any change bumps the rubric and
judge version together. Three binary metrics; every example shows the
verdict a correct grader must produce.

## 1. answer_correctness (binary)

Correct = the system's answer conveys the same essential information as
the gold answer. Wording, order, formatting free. Partial = incorrect.
Contradiction = incorrect. Abstention when gold exists = incorrect here
(abstention is graded separately).

Worked examples:

1. Q: "How many days did I spend in Japan and Chicago in total?"
   Gold: "12" | Answer: "You spent 12 days in total: 8 in Japan and 4 in
   Chicago." -> **correct** (same value, extra consistent detail).
2. Gold: "12" | Answer: "About 11 or 12 days." -> **incorrect** (does not
   commit to the gold value).
3. Gold: "Blue Bottle" | Answer: "blue bottle coffee" -> **correct**
   (case and suffix are formatting).
4. Gold: "May 2023" | Answer: "You moved in spring 2023, in May." ->
   **correct**.
5. Gold: "a 4-day trip to Chicago and an 8-day trip to Japan" |
   Answer: "You took a 4-day trip to Chicago." -> **incorrect** (partial:
   one required part missing).
6. Gold: "$150" | Answer: "It cost $150, although you later returned it."
   -> **incorrect** IF the return contradicts gold context; **correct** if
   the gold answer is only the price and the extra claim does not
   contradict it. Default when unsure whether an addition contradicts:
   grade on the gold-relevant content only -> **correct**.
7. Gold: "Rex" | Answer: "I don't know your dog's name." -> **incorrect**.
8. Gold: "twice a week" | Answer: "Two times per week." -> **correct**.

## 2. evidence_coverage (binary)

Covered = the information needed to produce the gold answer is present in
the retrieved set. Paraphrase counts; presence is judged over the whole
set; everything needed must be present; related-but-insufficient is not
coverage. Ignore what the system answered.

Worked examples:

1. Q: "Total days in Japan and Chicago?" Gold: "12" (8 + 4).
   Retrieved includes "User visited Japan from April 15 to April 22" and
   "User enjoyed Italian food during their 4-day Chicago trip". ->
   **covered** (both durations derivable, paraphrased).
2. Same question; retrieved includes only the Japan dates. ->
   **not covered** (one required fact missing).
3. Q: "What is my dog's name?" Gold: "Biscuit". Retrieved: "User adopted
   a golden retriever named Biscuit from the shelter." -> **covered**.
4. Same question; retrieved: "User adopted a dog from the shelter last
   weekend." -> **not covered** (right topic, needed fact absent).
5. Q: "Which laptop did I decide to buy?" Gold: "the ThinkPad".
   Retrieved contains the full verbatim conversation turn where the user
   says they will buy the ThinkPad. -> **covered** (verbatim is a special
   case of coverage).
6. Retrieved contains the fact stated and later corrected elsewhere in
   the set (old and new value both present). -> **covered** (presence, not
   resolution; update conflicts are a bucket, not a coverage failure).
7. Retrieved is empty. -> **not covered**.

## 3. abstention (binary)

Abstained = the answer declines to provide the information instead of
attempting an answer.

Worked examples:

1. "I don't know." -> **abstained**.
2. "The context does not mention your blood type." -> **abstained**.
3. "I believe it was 4 days." -> **not abstained** (commits to a value).
4. "It was 4 days, though I am not fully sure." -> **not abstained**.
5. "I cannot answer that from what I remember." -> **abstained**.
6. "There is no record of that; my guess would be Chicago." ->
   **not abstained** (a guess is an attempt).

## Two-sided abstention accuracy (derived, not judged directly)

Per item: on unanswerable items (the `abstention` stratum), correct
behavior is abstained=true; on answerable items, correct behavior is
abstained=false. Reported as one two-sided accuracy plus its two
components, per provider.
