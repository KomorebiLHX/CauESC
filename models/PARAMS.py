SAMPLE = False
TEMPERATURE = 1.0
OPTIM_STEP = 5000
ALPHA = 0.125
BETA = 0.1

STRATEGIES = {
    "Question": "[Question] Asking for information related to the problem to help the help-seeker articulate the is- sues that "
    "they face. Open-ended questions are best, and closed questions can be used to get specific information.",
    "Restatement or Paraphrasing": "[Restatement or Paraphrasing] A simple, more concise rephrasing of the help-seeker‘s "
    "statements that could help them see their situation more clearly.",
    "Reflection of feelings": "[Reflection of feelings] Articulate and describe the help-seeker’s feelings.",
    "Self-disclosure": "[Self-disclosure] Divulge similar experiences that you have had or emotions that you share with the help-seeker "
    "to express your empathy.",
    "Affirmation and Reassurance": "[Affirmation and Reassurance] Affirm the helpseeker’s strengths, motivation, and capabilities and provide "
    "reassurance and encouragement.",
    "Providing Suggestions": "[Providing Suggestions] Provide suggestions about how to change, but be careful to not overstep and tell them "
    "what to do.",
    "Information": "[Information] Provide useful information to the help-seeker, for example with data, facts, opinions, resources, "
    "or by answering questions.",
    "Others": "[Others] Exchange pleasantries and use other support strategies that do not fall into the above categories."
}

STRATEGY_LABELS = [
    "[Question]",
    "[Restatement or Paraphrasing]",
    "[Reflection of feelings]",
    "[Self-disclosure]",
    "[Affirmation and Reassurance]",
    "[Providing Suggestions]",
    "[Information]",
    "[Others]"
]