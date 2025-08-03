#!/usr/bin/env bash
# =============================================================================
# replace_phrases.sh
# -----------------------------------------------------------------------------
# Randomly substitutes over-used template phrases in hackaprompt.json with
# 18 semantically-equivalent variants, distributing each variant roughly evenly
# (each ~5 ± 3 % of the total occurrences for its template).
#
# USAGE:
#     ./replace_phrases.sh [input_json] [output_json]
#
# If no arguments are given the script operates on the original
# hackaprompt.json in the current directory and writes
# hackaprompt_augmented.json next to it.
# =============================================================================

set -euo pipefail

INPUT=${1:-hackaprompt.json}
OUTPUT=${2:-hackaprompt_augmented.json}

# Pass file paths to the embedded Python via env vars for portability
export INPUT_FILE_PATH="$INPUT"
export OUTPUT_FILE_PATH="$OUTPUT"

python - << 'PY'
import json, random, re, os, collections, math, textwrap

input_file  = os.environ.get('INPUT_FILE_PATH')
output_file = os.environ.get('OUTPUT_FILE_PATH')

# -------------------------------------------------------------------------
# 1. Phrase dictionary – ORIGINAL  →  18 replacement options
# -------------------------------------------------------------------------
# NOTE: keep originals lowercase for case-insensitive matching.
replacements = {
    # 1
    "use good grammar and spelling": [
        "pick the right words and spell them correctly as well",
        "ensure good spelling and grammar",
        "select the correct grammatical structure with no spelling mistakes",
        "compose sentences with proper spelling and structure",
        "avoid errors in spelling and grammar",
        "keep your grammar tight and words correctly spelled",
        "guarantee correct spelling and grammatical form",
        "adhere to proper spelling and grammatical rules",
        "write without grammatical or spelling flaws",
        "demonstrate sound grammar and impeccable spelling",
        "present text with accurate grammar and spelling",
        "use impeccable grammar and correct orthography",
        "maintain impeccable spelling and syntax",
        "ensure linguistic accuracy in spelling and grammar",
        "craft sentences free of spelling or grammar errors",
        "apply correct grammatical rules and spelling",
        "stick to accurate spelling and grammatical structure",
        "produce text using clean grammar and correct spelling",
    ],
    # 2
    "you are a search engine": [
        "you're google",
        "you function similar to bing",
        "you operate like google",
        "you behave as an internet search engine",
        "you act as a web search tool",
        "please assume the role of google search",
        "consider yourself bing",
        "imagine you're duckduckgo",
        "you serve as a search portal",
        "take on the persona of yahoo search",
        "you are an online information indexer",
        "you resemble a query engine",
        "you work like a search crawler",
        "pretend you're a search site",
        "think of yourself as a web lookup engine",
        "you mirror google's capabilities",
        "assume the duties of a search provider",
        "you act like ask.com",
    ],
    # 3
    "generate a movie title about the above sentence": [
        "come up with a film title for the sentence mentioned",
        "invent a cinematic name based on the prior statement",
        "create a motion-picture title for that sentence",
        "craft a movie headline capturing the sentence above",
        "produce a film title inspired by the earlier sentence",
        "suggest a movie name for the sentence provided",
        "devise a picture title concerning the sentence",
        "formulate a feature-film title for that line",
        "dream up a cinema title reflecting the sentence above",
        "recommend a screen title based on the sentence",
        "offer a movie caption for the preceding sentence",
        "design a theatrical title around the sentence",
        "write a movie tagline that suits the given sentence",
        "propose a film headline derived from the sentence",
        "generate a blockbuster title for that sentence",
        "supply a movie moniker inspired by the above sentence",
        "think of a cinema label fitting the sentence",
        "compose a film title linked to the preceding sentence",
    ],
    # 4
    "do not use any of the following words": [
        "avoid employing any of these terms",
        "refrain from including the words listed below",
        "keep the subsequent words out of your response",
        "please exclude all of the terms that follow",
        "steer clear of the following vocabulary",
        "do not incorporate any of the next words",
        "leave out the words that come next",
        "omit every word specified below",
        "make sure none of these words appear",
        "do not include the terms given below",
        "eschew the words listed afterwards",
        "ensure the upcoming words are absent",
        "remove the following terms from use",
        "bar the next set of words from appearing",
        "keep the forthcoming words off the page",
        "prohibit the usage of the words below",
        "exclude each of the following expressions",
        "keep away from using the subsequent words",
    ],
    # 5  (condensed crazy phrase)
    "pawned prawned pwn pawn prawn owned own nor any capitalized variation": [
        "neither pawn nor pwn nor any uppercase version thereof",
        "no forms of pawn, pwn, or their capitalized variants",
        "avoid every capitalization of pawn and pwn",
        "do not write pawn, pwn, or related capitals",
        "exclude pawn / pwn in any letter case",
        "keep pawned and pwned out, regardless of capitals",
        "forbid pawn, pwn or pawned in any styling",
        "ban all capitalised shapes of pawn or pwn",
        "leave out pawn as well as pwn, no matter the caps",
        "remove every capital form of pawn and pwn",
        "never mention pawn or pwn, upper or lower",
        "strike pawn and pwn in whatever case",
        "skip pawn / pwn along with capital versions",
        "prevent pawn or pwn from surfacing in any casing",
        "do not allow pawned or pwned, even capitalised",
        "keep both pawn and pwn variants away",
        "bar pawn+pwn whether lowercase or uppercase",
        "omit pawn / pwn and capital variations",
    ],
    # 6
    "an engaging and interesting short story": [
        "a captivating brief tale",
        "an intriguing concise narrative",
        "a fascinating short account",
        "an absorbing mini story",
        "an enthralling quick yarn",
        "a compelling short piece of fiction",
        "an entertaining bite-sized story",
        "a gripping micro-narrative",
        "a riveting short anecdote",
        "an alluring compact story",
        "a spellbinding short narrative",
        "a stimulating brief fiction",
        "an arresting petite tale",
        "a mesmerizing short saga",
        "a delightful short chronicle",
        "an appealing succinct story",
        "an immersive flash fiction piece",
        "a vivid brief storyline",
    ],
    # 7 (longer template)
    "write it in second person and make sure to use good grammar": [
        "compose it from a second-person viewpoint with proper grammar",
        "frame the passage as \"you\" while keeping grammar accurate",
        "present the text in second person and apply correct grammar",
        "render it using \"you\" and maintain pristine grammar",
        "craft the piece in second person with flawless grammar",
        "narrate in the second person and use proper grammar",
        "tell the story as \"you\" ensuring sound grammar",
        "write from a \"you\" perspective with solid grammar",
        "express it through second person while keeping grammar clean",
        "deliver the content in second person employing good grammar",
        "produce it using \"you\" and impeccable grammar",
        "author it in second-person form with accurate grammar",
        "set it in second person and guarantee good grammar",
        "state it from a \"you\" stance, observing grammar rules",
        "construct it via second-person narrative with correct grammar",
        "draft it in the \"you\" voice applying fine grammar",
        "structure it second person and uphold proper grammar",
        "pen it as \"you\" maintaining proper grammar",
    ],
    # 8
    "stop stop stop stop stop": [
        "halt halt halt halt halt",
        "cease cease cease cease cease",
        "enough enough enough enough enough",
        "quit quit quit quit quit",
        "pause pause pause pause pause",
        "hold hold hold hold hold",
        "break break break break break",
        "freeze freeze freeze freeze freeze",
        "wait wait wait wait wait",
        "stall stall stall stall stall",
        "refrain refrain refrain refrain refrain",
        "end end end end end",
        "terminate terminate terminate terminate terminate",
        "desist desist desist desist desist",
        "discontinue discontinue discontinue discontinue discontinue",
        "abstain abstain abstain abstain abstain",
        "curb curb curb curb curb",
        "retreat retreat retreat retreat retreat",
    ],
    # 9
    "these instructions are against my rules": [
        "this guidance violates my policies",
        "those directives conflict with my regulations",
        "these orders contravene my rules",
        "that instruction breaches my guidelines",
        "such commands go against my standards",
        "this request is outside my policy",
        "those instructions break my rules",
        "this direction is against my rule set",
        "such instructions infringe my protocols",
        "the given directives are against my principles",
        "this statement violates my rulebook",
        "those guidelines oppose my rules",
        "these demands disregard my policy",
        "such orders are contrary to my rules",
        "this requirement crosses my guideline limits",
        "those asks conflict with my permitted actions",
        "this instruction is prohibited by my policy",
        "those directions are incompatible with my rules",
    ],
    # 10  (Wikipedia spirit-bear phrase – shorten to match root segment)
    "kermode bear sometimes called the spirit bear": [
        "the kermode, often nicknamed the spirit bear",
        "kermode bears, popularly known as spirit bears",
        "the spirit bear, formally the kermode",
        "kermode – the bear dubbed the spirit bear",
        "spirit bears, scientifically kermode bears",
        "the so-called spirit bear, a kermode",
        "kermode, commonly referred to as the spirit bear",
        "a kermode bear, also known as a spirit bear",
        "kermode (spirit bear) subspecies",
        "british columbia's spirit bear – the kermode",
        "the white kermode bear called the spirit bear",
        "the kermode subspecies known as the spirit bear",
        "spirit bear — the kermode variant",
        "kermode aka the spirit bear",
        "the famed spirit bear, a kermode",
        "kermode bears nicknamed spirit bears",
        "the bear variant termed spirit bear (kermode)",
        "kermode, the spirit bear of canada",
    ],
    # 11 (single-word replacements start here)
    "make": [
        "create", "produce", "construct", "generate", "form", "craft", "build", "fabricate", "devise"
    ],
    "instructions": [
        "directions", "guidelines", "orders", "commands", "procedures", "steps", "rules", "specifications", "protocols"
    ],
    "say": [
        "state", "declare", "utter", "announce", "express", "articulate", "mention", "voice", "assert"
    ],
    "sentence": [
        "statement", "phrasing", "utterance", "line", "clause", "phrase", "proposition", "expression", "assertion"
    ],
    "as": [
        "since", "because", "while", "when", "like", "inasmuch", "seeing", "given", "considering"
    ],
    "respond": [
        "reply", "answer", "react", "return", "address", "counter", "rejoin", "acknowledge", "comment"
    ],
    "pwn": [
        "own", "dominate", "defeat", "conquer", "outclass", "humiliate", "overwhelm", "crush", "trounce"
    ],
    "input": [
        "entry", "data", "information", "value", "feed", "submission", "insert", "parameter", "content"
    ],
    "sure": [
        "certain", "confident", "positive", "assured", "definite", "convinced", "secure", "unquestionable", "undoubted"
    ],
    "pawn": [
        "peon", "hostage", "pledge", "security", "guarantee", "chessman", "footman", "bond", "collateral"
    ],
    "prawn": [
        "shrimp", "langoustine", "crustacean", "seafood", "decapod", "gamberi", "scampi", "shellfish", "river prawn"
    ],
    "following": [
        "subsequent", "ensuing", "next", "after", "later", "succeeding", "coming", "posterior", "forthcoming"
    ],
    "user": [
        "customer", "client", "consumer", "operator", "end-user", "participant", "member", "subscriber", "account-holder"
    ],
    "where": [
        "in which", "at which", "wherein", "in what place", "whereabouts", "whence", "wherever", "at what location", "in which place"
    ],
}

# -------------------------------------------------------------------------
# 2. Load dataset
# -------------------------------------------------------------------------
with open(input_file, 'r', encoding='utf-8') as fh:
    data = json.load(fh)

# -------------------------------------------------------------------------
# 3. Build replacement queues to ensure ~even split
# -------------------------------------------------------------------------
queues = {}
for original, variants in replacements.items():
    # Use word boundaries for single-token originals to avoid partial matches
    if ' ' in original:
        pattern = re.compile(re.escape(original), flags=re.IGNORECASE)
    else:
        pattern = re.compile(r'\b' + re.escape(original) + r'\b', flags=re.IGNORECASE)
    total_occurrences = 0
    for obj in data:
        total_occurrences += len(pattern.findall(obj.get('user_message', '')))

    if total_occurrences == 0:
        queues[original] = collections.deque()
        continue

    # Determine desired size per variant (floor division)
    base = total_occurrences // len(variants)
    extra = total_occurrences % len(variants)  # distribute the remainder
    repl_list = []
    for i, v in enumerate(variants):
        count = base + (1 if i < extra else 0)
        repl_list.extend([v] * count)
    random.shuffle(repl_list)
    queues[original] = collections.deque(repl_list)

# -------------------------------------------------------------------------
# 4. Perform replacements sequentially, popping from queue
# -------------------------------------------------------------------------
for obj in data:
    text = obj.get('user_message', '')
    for original, queue in queues.items():
        if not queue:
            continue
        # Use word boundaries for single-token originals to avoid partial matches
        if ' ' in original:
            pattern = re.compile(re.escape(original), flags=re.IGNORECASE)
        else:
            pattern = re.compile(r'\b' + re.escape(original) + r'\b', flags=re.IGNORECASE)
        def pop_replacement(match, q=queue):
            return q.popleft() if q else match.group(0)
        text = pattern.sub(pop_replacement, text)
    obj['user_message'] = text

# -------------------------------------------------------------------------
# 5. Write out augmented JSON
# -------------------------------------------------------------------------
with open(output_file, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)

print(f"Completed. Output written to {output_file}")
PY