import pickle
import pandas as pd

GOLDEN_TRUTH = False
MAX_SEQ_LEN = 128
WINDOW = 10

CAUSAL_DF = pd.read_csv("_reformat/evaluation_esconv_classification_save_preds.csv", index_col=3)
EMOTIONS = [
    'anxiety', 'depression', 'sadness', 'anger', 'fear', 'shame', 'disgust', 'nervousness', 'pain', 'jealousy', 'guilt'
]

COMET_FEATURES = pickle.load(open('_reformat/esconv_features_comet.pkl', 'rb'))
COMET_RELATIONS = ["xIntent", "xAttr", "xNeed", "xWant", "xEffect", "xReact", "oWant", "oEffect", "oReact"]
PERSONA_RELATIONS = ["xIntent", "xAttr", "xNeed", "xWant", "xEffect", "xReact"]
SEK_RELATIONS = ["xWant", "xEffect", "xReact"]
SUP_RELATIONS = ["oWant", "oEffect", "oReact"]

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
