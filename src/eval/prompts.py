# src/eval/prompts.py
"""
Prompt templates for LLM-as-judge scorers.

Following best practices from research:
- Binary pass/fail with critique (Hamel Husain)
- Claim extraction + verification (Ragas Faithfulness)
- Clear, specific instructions
"""

# =============================================================================
# FAITHFULNESS PROMPTS
# =============================================================================

EXTRACT_CLAIMS_PROMPT = """\
Given a question and an answer, extract all factual claims made in the answer.

WHAT TO EXTRACT:
- Factual statements about the law, regulation, or legal requirements
- Statements citing specific articles, paragraphs, or provisions
- Conclusions or recommendations based on legal analysis

WHAT TO SKIP (do not extract as claims):
- Meta-statements like "I cannot conclude..." or "More information is needed..."
- Structural statements like "The requirements depend on..."
- Questions or clarification requests
- Citation markers like "[1]", "[2]" on their own

Question: {question}

Answer: {answer}

Extract each distinct FACTUAL claim as a separate item. Be comprehensive but avoid duplicates.
Output as a JSON array of strings.

Example output:
["The AI Act was adopted in 2024", "High-risk AI systems require conformity assessment", "Article 6 defines high-risk classification"]

Claims:
"""

VERIFY_CLAIMS_PROMPT = """\
You are verifying if claims from an answer are supported by the provided context.

IMPORTANT DISTINCTIONS:
1. FACTUAL CLAIMS about the law/regulation → Verify against context
2. META-STATEMENTS about the answer itself → Mark as SUPPORTED (these are not factual claims)
3. CONDITIONAL STATEMENTS ("If X, then Y") → The conditional logic should be verifiable
4. STATEMENTS about missing user information → Mark as SUPPORTED (legitimate clarification requests)

Examples of META-STATEMENTS to mark as SUPPORTED:
- "I cannot conclude classification based solely on the found sources" → SUPPORTED (appropriate epistemic humility)
- "The requirements depend on the classification" → SUPPORTED (true structural statement)
- "More information is needed to determine..." → SUPPORTED (legitimate clarification)

A FACTUAL CLAIM is SUPPORTED if:
- The context contains information that directly supports or implies it
- It is a meta-statement about the answer's limitations or conditions

A FACTUAL CLAIM is NOT SUPPORTED if:
- The context contradicts it
- It makes a specific factual assertion about the law that cannot be verified from context

Context:
{context}

For each claim below, respond with "SUPPORTED" or "NOT_SUPPORTED" and a brief explanation.

Claims to verify:
{claims}

Respond in JSON format:
{{
    "verifications": [
        {{"claim": "...", "verdict": "SUPPORTED" or "NOT_SUPPORTED", "explanation": "..."}}
    ]
}}
"""

# =============================================================================
# ANSWER RELEVANCY PROMPTS
# =============================================================================

ANSWER_RELEVANCY_PROMPT = """\
You are evaluating if an answer adequately addresses the user's question.

IMPORTANT: For legal/compliance questions, a GOOD answer may:
1. Decline to give definitive advice when insufficient information is provided
2. Request clarification about the user's specific situation
3. Explain that classification "depends on" specific factors
4. Refuse to make claims "without reservation" if the law has exceptions

These responses are RELEVANT and APPROPRIATE - they address the question correctly by:
- Acknowledging what can and cannot be determined
- Providing the applicable legal framework
- Explaining conditions and dependencies

Question: {question}

Answer: {answer}

Evaluate on a scale of 0-10:
- 10: Answer fully addresses the question OR appropriately declines with explanation
- 7-9: Answer addresses the question with minor gaps, or provides helpful context while declining
- 4-6: Answer partially addresses the question or is tangential
- 1-3: Answer barely relates to the question
- 0: Answer does not address the question at all

Respond in JSON format:
{{
    "score": <0-10>,
    "critique": "<detailed explanation of your rating>"
}}
"""

# =============================================================================
# GROUNDEDNESS PROMPT (alternative to Faithfulness)
# =============================================================================

GROUNDEDNESS_PROMPT = """\
You are evaluating if an answer is grounded in the provided context.
An answer is grounded if all its claims can be traced back to the context.

Context:
{context}

Question: {question}

Answer: {answer}

Evaluate:
1. Are all factual claims in the answer supported by the context?
2. Does the answer contain any information not found in the context (hallucination)?
3. Does the answer contradict any information in the context?

Respond in JSON format:
{{
    "grounded": true or false,
    "score": <0.0-1.0>,
    "unsupported_claims": ["list of claims not supported by context"],
    "critique": "<detailed explanation>"
}}
"""
