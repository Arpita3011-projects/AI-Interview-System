"""
Post-interview analysis helpers using Gemini text models.

This module runs AFTER the live voice interview has finished, using
stored interview metadata to generate a rich evaluation JSON including:
- communication skills
- emotion analysis
- filler word usage
- grammar quality
- sentiment
- technical depth / correctness / relevance
"""

import logging
from typing import Any, Dict

from django.conf import settings

logger = logging.getLogger(__name__)


def _build_analysis_prompt(interview) -> str:
    """
    Build a structured prompt for Gemini to analyse the interview.

    Notes:
    - We currently don't persist a full verbatim transcript of the
      candidate's answers, but we DO have:
        * candidate resume summary
        * experience level
        * questions asked
        * cheating events summary (if any)
        * average voice confidence score from audio processing
      Gemini should infer a plausible, conservative evaluation based on
      this metadata and typical interview behaviour.
    """
    resume = interview.resume
    skills = resume.skills if isinstance(resume.skills, list) else []
    experience_level = interview.experience_level

    avg_voice_conf = getattr(interview, "average_voice_confidence", 0.0) or 0.0
    cheating_summary = interview.cheating_events_summary or []
    questions = interview.questions_asked or []

    return f"""
You are an expert technical interviewer and communication coach.
You are given structured METADATA about a completed AI-led interview.
From this metadata you must produce a conservative, realistic assessment
of the candidate's performance.

IMPORTANT CONSTRAINTS:
- You DO NOT have the exact transcript of candidate answers.
- However, assume the AI interviewer followed the scripted protocol and
  asked the questions listed in the metadata.
- The average_voice_confidence value comes from detailed audio analysis
  (pitch stability, tremor, pauses, speaking speed).
- Cheating events summary indicates integrity issues, if any.

Be realistic and slightly strict in scoring. Do NOT invent specific
quotes; describe behaviour in general terms only.

=== METADATA ===
Candidate name: {resume.candidate_name or 'Unknown'}
Experience level: {experience_level}
Key skills (from resume): {', '.join(skills[:15]) if skills else 'Not specified'}
Average voice confidence (0-1): {avg_voice_conf:.3f}
Cheating events summary (JSON): {cheating_summary}
Questions asked (JSON list): {questions}

=== OUTPUT FORMAT (STRICT JSON) ===
Return ONLY a JSON object (no markdown, no comments) with this shape:
{{
  "communication": {{
    "score_0_100": 0,
    "justification": "",
    "filler_words": {{
      "total_count_estimate": 0,
      "density_per_min_estimate": 0.0,
      "examples": []
    }},
    "grammar": {{
      "score_0_100": 0,
      "issues": []
    }},
    "speaking_speed": {{
      "qualitative": "slow|balanced|fast",
      "comment": ""
    }},
    "pauses": {{
      "frequency": "low|medium|high",
      "comment": ""
    }}
  }},
  "emotion": {{
    "dominant": "calm|nervous|confident|enthusiastic|frustrated|mixed",
    "confidence_0_1": 0.0,
    "notes": ""
  }},
  "sentiment": {{
    "overall": "positive|neutral|negative",
    "confidence_0_1": 0.0,
    "notes": ""
  }},
  "technical": {{
    "depth_score_0_100": 0,
    "correctness_score_0_100": 0,
    "relevance_score_0_100": 0,
    "summary": "",
    "strengths": [],
    "gaps": []
  }},
  "overall_rating": 0,
  "summary": "",
  "strengths": [],
  "areas_for_improvement": []
}}

SCORING GUIDELINES (COMMUNICATION):
- 90-100: Outstanding, very clear and structured, strong examples.
- 70-89: Good, mostly clear, minor issues.
- 50-69: Mixed, noticeable issues but understandable overall.
- 30-49: Weak, frequent clarity / structure problems.
- 0-29: Very poor communication.

SCORING GUIDELINES (TECHNICAL):
- Depth: how far beyond surface definitions they could reasonably go.
- Correctness: how often answers would be factually / conceptually correct.
- Relevance: how well they would stay on-topic w.r.t. questions & skills.

Make sure the JSON is valid and strictly follows the schema keys.
"""


def run_interview_text_analysis(interview) -> Dict[str, Any]:
    """
    Run post-interview text analysis using Gemini and store results on the Interview.

    Returns the evaluation dict (possibly empty if Gemini is not configured
    or an error occurs).
    """
    if not settings.GEMINI_API_KEY:
        logger.warning("Gemini API key not configured; skipping interview analysis.")
        return {}

    try:
        import google.generativeai as genai
        import json
    except Exception as e:  # pragma: no cover - optional dependency
        logger.error(f"Failed to import google.generativeai: {e}")
        return {}

    try:
        genai.configure(api_key=settings.GEMINI_API_KEY)
        # Use a fast text model for analysis
        model_name = "gemini-2.0-flash-001"
        model = genai.GenerativeModel(model_name)

        prompt = _build_analysis_prompt(interview)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()

        # Some models occasionally wrap JSON in markdown fences; strip them.
        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        evaluation = json.loads(text)

        # Merge into existing evaluation (if any)
        current_eval = interview.evaluation or {}
        if not isinstance(current_eval, dict):
            current_eval = {}
        current_eval.update(evaluation)

        interview.evaluation = current_eval
        interview.save(update_fields=["evaluation", "updated_at"])

        logger.info(f"Interview analysis completed for {interview.id}")
        return evaluation

    except Exception as e:
        logger.error(f"Interview analysis with Gemini failed: {e}", exc_info=True)
        return {}

