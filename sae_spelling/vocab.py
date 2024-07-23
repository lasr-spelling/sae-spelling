from collections import defaultdict
from typing import Callable

import nltk
from nltk.corpus import brown, words
from transformers import PreTrainedTokenizerFast

LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTERS_UPPER = LETTERS.upper()
ALL_ALPHA_LETTERS = LETTERS + LETTERS_UPPER


def get_tokens(
    tokenizer: PreTrainedTokenizerFast,
    filter: Callable[[str], bool] = lambda _token: True,
    replace_special_chars: bool = False,
) -> list[str]:
    result = []
    for token in tokenizer.vocab.keys():
        word = tokenizer.convert_tokens_to_string([token])
        if filter(word):
            result.append(word if replace_special_chars else token)
    return result


def get_alpha_tokens(
    tokenizer: PreTrainedTokenizerFast,
    allow_leading_space: bool = True,
    replace_special_chars: bool = False,
) -> list[str]:
    def filter_alpha(token: str) -> bool:
        if allow_leading_space and token.startswith(" "):
            token = token[1:]
        if len(token) == 0:
            return False
        return all(char in ALL_ALPHA_LETTERS for char in token)

    return get_tokens(
        tokenizer, filter_alpha, replace_special_chars=replace_special_chars
    )


def get_common_word_tokens(
    tokenizer: PreTrainedTokenizerFast,
    only_start_of_word: bool = True,
    replace_special_chars: bool = False,
    threshold: int = 10,
) -> list[str]:
    common_words = set(get_common_words(threshold=threshold).keys())

    def filter_common_words(token: str) -> bool:
        if only_start_of_word and not token.startswith(" "):
            return False
        if token.startswith(" "):
            token = token[1:]
        return token in common_words

    return get_tokens(
        tokenizer, filter_common_words, replace_special_chars=replace_special_chars
    )


def get_nltk_words() -> list[str]:
    try:
        return words.words()
    except LookupError:
        nltk.download("words")
    return words.words()


def get_brown_words() -> list[str]:
    # Useful for getting some measure of occurrence of words in the English language
    try:
        return brown.words()
    except LookupError:
        nltk.download("brown")
    return brown.words()


def get_common_words(threshold=10) -> dict[str, int]:
    """
    Get frequent English words that occur in the Brown corpus.
    """
    all_words = get_nltk_words()

    # Get word frequencies
    word_freq = defaultdict(int)
    for word in get_brown_words():
        word_freq[word.lower()] += 1

    # Filter words based on frequency
    return {
        word: word_freq[word.lower()]
        for word in all_words
        if word_freq[word.lower()] >= threshold
    }


def group_words_by_ending_and_length(
    words: list[str],
) -> dict[int, dict[str, list[str]]]:
    """
    Makes groups of words that start with a different letter, but have the same ending.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for word in words:
        if len(word) > 1:
            groups[len(word)][word[1:]].append(word)

    # Filter out groups with only one word
    return {
        length: {
            ending: words for ending, words in length_group.items() if len(words) > 1
        }
        for length, length_group in groups.items()
    }


def get_same_ending_word_pairs_of_len(length: int) -> list[tuple[str, str]]:
    """
    Get a list of word tuples that have the same length and ending.
    E.g. [("union", "onion"), ("roman", "woman")]
    """
    common_words_lowercase = {word.lower() for word in get_common_words().keys()}

    word_groups = group_words_by_ending_and_length(list(common_words_lowercase))

    pairs = []

    for similar_words in word_groups[length].values():
        pairs.append((similar_words[0], similar_words[1]))

    return pairs


# From Claude
SAMPLE_WORD_DEFINITIONS = {
    "alice": "A female given name, famously associated with the protagonist in Lewis Carroll's 'Alice's Adventures in Wonderland'.",
    "allen": "A male given name of English origin, meaning 'little rock' or 'harmony'.",
    "aloud": "In a voice loud enough to be heard.",
    "arose": "Past tense of arise, meaning to get up or come into existence.",
    "badly": "In a manner that is of poor quality or unpleasant.",
    "baker": "A person who makes bread, cakes, and other baked goods as a profession.",
    "beach": "A pebbly or sandy shore, especially by the sea between high- and low-water marks.",
    "billy": "A club or baton carried by police officers.",
    "blame": "To assign responsibility for a fault or wrong.",
    "block": "A solid piece of hard material, especially rock, stone, or wood, typically with flat surfaces on each side.",
    "blood": "The red liquid that circulates in the arteries and veins of humans and other vertebrate animals.",
    "bloom": "A flower or a period of flowering.",
    "bobby": "A British police officer.",
    "bound": "Going to; intending to go to.",
    "brace": "A device used to support or connect things or to strengthen a person's body.",
    "brain": "An organ of soft nervous tissue contained in the skull of vertebrates, functioning as the coordinating center of sensation and intellectual and nervous activity.",
    "brand": "A type of product manufactured by a particular company under a particular name.",
    "brass": "A yellow alloy of copper and zinc.",
    "brave": "Ready to face and endure danger or pain; showing courage.",
    "breed": "A particular strain or type of animal or plant.",
    "brick": "A small rectangular block typically made of fired or sun-dried clay, used in building.",
    "bride": "A woman on her wedding day or just before and after the event.",
    "brief": "Of short duration; not lasting for long.",
    "brown": "Of a color produced by mixing red, yellow, and black, as of dark wood or rich soil.",
    "buddy": "A close friend.",
    "built": "Past tense and past participle of build, meaning constructed or created.",
    "bunch": "A number of things, typically of the same kind, growing or fastened together.",
    "candy": "A sweet food made with sugar or syrup combined with fruit, chocolate, or nuts.",
    "carry": "To support and move (someone or something) from one place to another.",
    "catch": "To intercept and hold (something that has been thrown, propelled, or dropped).",
    "cause": "A person or thing that gives rise to an action, phenomenon, or condition.",
    "cease": "To come or bring to an end.",
    "chase": "To pursue in order to catch or catch up with.",
    "chose": "Past tense of choose, meaning to pick out or select (someone or something) as being the best or most appropriate of two or more alternatives.",
    "class": "A set or category of things having some property or attribute in common and differentiated from others by kind, type, or quality.",
    "clock": "A mechanical or electrical device for measuring time, indicating hours, minutes, and sometimes seconds.",
    "cloud": "A visible mass of condensed water vapor floating in the atmosphere, typically high above the ground.",
    "clung": "Past tense and past participle of cling, meaning to hold on tightly to.",
    "coast": "The part of the land adjoining or near the sea.",
    "couch": "A long upholstered piece of furniture for several people to sit on.",
    "could": "Past tense of can, used as a modal verb to indicate possibility or ability.",
    "count": "To determine the total number of (a collection of items).",
    "cover": "To put something on top of or in front of (something), especially in order to protect or conceal it.",
    "crack": "A line on the surface of something along which it has split without breaking into separate parts.",
    "craft": "An activity involving skill in making things by hand.",
    "cream": "The thick white or pale yellow fatty liquid that rises to the top of milk.",
    "creek": "A stream, brook, or minor tributary of a river.",
    "cried": "Past tense and past participle of cry, meaning to shed tears, typically as an expression of distress, pain, or sorrow.",
    "crime": "An action or omission that constitutes an offense that may be prosecuted by the state and is punishable by law.",
    "cross": "A mark, object, or figure formed by two intersecting lines at right angles.",
    "crown": "A circular ornamental headdress worn by a monarch as a symbol of authority.",
    "curse": "A solemn utterance intended to invoke a supernatural power to inflict harm or punishment on someone or something.",
    "dandy": "An excessively or foppishly dressed man.",
    "ditch": "A narrow channel dug in the ground, typically used for drainage alongside a road or the edge of a field.",
    "dodge": "To avoid (someone or something) by a sudden quick movement.",
    "doing": "Present participle of do, meaning to perform or execute an action.",
    "dough": "A thick, malleable mixture of flour and liquid, used for baking into bread or pastry.",
    "draft": "A preliminary version of a piece of writing.",
    "drain": "To draw off (liquid) gradually from; to empty or dry.",
    "drank": "Past tense of drink, meaning to take (a liquid) into the mouth and swallow.",
    "dream": "A series of thoughts, images, and sensations occurring in a person's mind during sleep.",
    "dress": "A one-piece garment for a woman or girl that covers the body and extends down over the legs.",
    "dried": "Past tense and past participle of dry, meaning to remove moisture from.",
    "drill": "A tool or machine with a rotating cutting tip or reciprocating hammer or chisel, used for making holes.",
    "drove": "Past tense of drive, meaning to operate and control the direction and speed of a motor vehicle.",
    "dying": "Present participle of die, meaning to stop living.",
    "eight": "A cardinal number, one more than seven or two less than ten.",
    "elder": "Of a greater age; older.",
    "ellen": "A female given name, often used as a shortened form of Eleanor or Helen.",
    "faced": "Past tense and past participle of face, meaning to confront and deal with or accept.",
    "faint": "Lacking strength or vigor; weak.",
    "fence": "A barrier, railing, or other upright structure, typically of wood or wire, enclosing an area of ground to mark a boundary, control access, or prevent escape.",
    "ferry": "A boat or ship for transporting passengers and often vehicles across a body of water.",
    "fever": "An abnormally high body temperature, usually accompanied by shivering, headache, and in severe instances, delirium.",
    "field": "An area of open land, especially one planted with crops or pasture.",
    "fight": "To take part in a violent struggle involving the exchange of physical blows or the use of weapons.",
    "fired": "Past tense and past participle of fire, meaning to discharge a gun or other weapon.",
    "fixed": "Fastened securely in position.",
    "flame": "A hot glowing body of ignited gas that is generated by something on fire.",
    "flock": "A number of birds of one kind feeding, resting, or traveling together.",
    "flood": "An overflowing of a large amount of water beyond its normal confines, especially over what is normally dry land.",
    "flung": "Past tense and past participle of fling, meaning to throw or hurl forcefully.",
    "forth": "Out from a starting point and forward or into view.",
    "found": "Past tense and past participle of find, meaning to discover or perceive by chance or unexpectedly.",
    "frank": "Open, honest, and direct in speech or writing, especially when dealing with unpalatable matters.",
    "freed": "Past tense and past participle of free, meaning to release from captivity, confinement, or slavery.",
    "funny": "Causing laughter or amusement; humorous.",
    "glass": "A hard, brittle substance, typically transparent or translucent, made by fusing sand with soda, lime, and sometimes other ingredients.",
    "gloom": "Partial or total darkness, especially when regarded as dreary.",
    "going": "Present participle of go, meaning to move or travel.",
    "grace": "Smoothness and elegance of movement.",
    "grade": "A particular level of rank, quality, proficiency, intensity, or value.",
    "grain": "A small, hard seed, especially the seed of a food plant such as wheat, corn, rye, oats, rice, or millet.",
    "grand": "Magnificent and imposing in appearance, size, or style.",
    "grass": "Vegetation consisting of typically short plants with long, narrow leaves, growing wild or cultivated on lawns and pasture.",
    "grave": "A place of burial for a dead body, typically a hole dug in the ground and marked by a stone or mound.",
    "great": "Of an extent, amount, or intensity considerably above the normal or average.",
    "greek": "Relating to Greece, its people, or their language.",
    "grief": "Deep sorrow, especially that caused by someone's death.",
    "grill": "A device on which food is broiled or grilled.",
    "gross": "Unattractively large or bloated.",
    "grove": "A small wood or other group of trees.",
    "grown": "Past participle of grow, meaning to increase in size or amount.",
    "guest": "A person who is invited to visit someone's home or attend a particular social occasion.",
    "guilt": "The fact of having committed a specified or implied offense or crime.",
    "handy": "Convenient to handle or use; useful.",
    "harry": "A male given name, often a diminutive of Harold or Henry.",
    "hence": "As a consequence; for this reason.",
    "hired": "Past tense and past participle of hire, meaning to employ for wages.",
    "honey": "A sweet, sticky yellowish-brown fluid made by bees and other insects from nectar collected from flowers.",
    "horse": "A large plant-eating domesticated mammal with solid hoofs and a flowing mane and tail, used for riding, racing, and to carry and pull loads.",
    "hotel": "An establishment providing accommodation, meals, and other services for travelers and tourists.",
    "house": "A building for human habitation, especially one that is lived in by a family or small group of people.",
    "humor": "The quality of being amusing or comic, especially as expressed in literature or speech.",
    "jerry": "A male given name, often a diminutive of Jerome or Gerald.",
    "joint": "A point at which parts of an artificial structure are joined.",
    "later": "At a time in the near future; after the time of writing or speaking.",
    "lease": "A contract by which one party conveys land, property, services, etc. to another for a specified time, usually in return for a periodic payment.",
    "lever": "A rigid bar resting on a pivot, used to move a heavy or firmly fixed load with one end when pressure is applied to the other.",
    "light": "The natural agent that stimulates sight and makes things visible.",
    "liver": "A large lobed glandular organ in the abdomen of vertebrates, involved in many metabolic processes.",
    "lobby": "A room in a public building used for entrance from the outside.",
    "local": "Belonging or relating to a particular area or neighborhood, typically exclusively so.",
    "lodge": "A small house at the gates of a park or in the grounds of a large house, occupied by a gatekeeper, gardener, or other employee.",
    "lover": "A partner in a sexual or romantic relationship.",
    "lower": "To move (something) in a downward direction.",
    "loyal": "Giving or showing firm and constant support or allegiance to a person or institution.",
    "lunch": "A meal eaten in the middle of the day, typically one that is lighter or less formal than an evening meal.",
    "lying": "Present participle of lie, meaning to be in or assume a horizontal or resting position on a supporting surface.",
    "maker": "A person or thing that makes or produces something.",
    "maris": "A female given name, variant of Mary.",
    "marry": "To enter into a formal union with (someone) as a spouse according to law or custom.",
    "marty": "A male given name, often a diminutive of Martin.",
    "match": "A contest in which people or teams compete against each other in a particular sport.",
    "might": "Used to express possibility or make a suggestion.",
    "mixed": "Consisting of different qualities or elements.",
    "money": "A current medium of exchange in the form of coins and banknotes; coins and banknotes collectively.",
    "motel": "A roadside hotel designed primarily for motorists, typically having rooms arranged in low blocks with parking directly outside.",
    "mound": "A rounded mass projecting above a surface.",
    "mount": "To climb up (stairs, a hill, or other rising surface).",
    "mouse": "A small rodent that typically has a pointed snout, relatively large ears and eyes, and a long tail.",
    "mouth": "The opening in the lower part of the human face, surrounded by the lips, through which food is taken in and vocal sounds are emitted.",
    "muddy": "Covered in, full of, or resembling mud.",
    "nerve": "A whitish fiber or bundle of fibers in the body that transmits impulses of sensation to the brain or spinal cord, and impulses from these to the muscles and organs.",
    "never": "At no time in the past or future; on no occasion; not ever.",
    "night": "The period of darkness in each twenty-four hours; the time from sunset to sunrise.",
    "north": "The direction or point on the mariner's compass at 0° or 360°, corresponding to the northward cardinal direction.",
    "nurse": "A person trained to care for the sick or infirm, especially in a hospital.",
    "older": "Having lived for a relatively long time; not young.",
    "onion": "An edible bulb with a pungent taste and smell, composed of several concentric layers, used in cooking.",
    "paced": "Past tense and past participle of pace, meaning to walk at a steady speed, especially without a particular destination and as an expression of anxiety or annoyance.",
    "paint": "A colored substance which is spread over a surface and dries to leave a thin decorative or protective coating.",
    "paris": "The capital and most populous city of France.",
    "party": "A social gathering of invited guests, typically involving eating, drinking, and entertainment.",
    "paste": "A thick, soft, moist substance typically produced by mixing dry ingredients with a liquid.",
    "patch": "A small area of something different from its surroundings.",
    "pause": "A temporary stop in action or speech.",
    "phase": "A distinct period or stage in a process of change or forming part of something's development.",
    "pitch": "To throw or toss.",
    "plate": "A flat dish, typically circular and made of china, from which food is eaten or served.",
    "point": "A particular spot, place, or position in an area or on a map, object, or surface.",
    "pound": "A unit of weight in general use equal to 16 oz. avoirdupois (0.4536 kg).",
    "power": "The ability or capacity to do something or act in a particular way.",
    "press": "To move or cause to move into a position of contact with something by exerting continuous physical force.",
    "pride": "A feeling of deep pleasure or satisfaction derived from one's own achievements, the achievements of those with whom one is closely associated, or from qualities or possessions that are widely admired.",
    "prime": "Of first importance; main.",
    "prose": "Written or spoken language in its ordinary form, without metrical structure.",
    "prove": "To demonstrate the truth or existence of (something) by evidence or argument.",
    "purse": "A small bag used especially by women to carry money and personal belongings.",
    "quest": "A long or arduous search for something.",
    "quite": "To the utmost or most absolute extent or degree; absolutely; completely.",
    "rally": "A mass meeting of people making a political protest or showing support for a cause.",
    "reach": "To stretch out an arm in order to touch or grasp something.",
    "right": "Morally good, justified, or acceptable.",
    "river": "A large natural stream of water flowing in a channel to the sea, a lake, or another such stream.",
    "roast": "To cook (food, especially meat) by prolonged exposure to heat in an oven or over a fire.",
    "roman": "Relating to or characteristic of Rome or the Roman Empire.",
    "rough": "Having an uneven or irregular surface; not smooth or level.",
    "round": "Shaped like or approximately like a circle or cylinder.",
    "royal": "Having the status of or connected with a king or queen.",
    "sadly": "In a manner that shows sorrow or regret.",
    "saint": "A person acknowledged as holy or virtuous and typically regarded as being in heaven after death.",
    "sally": "A sudden charge out of a besieged place against the enemy; a sortie.",
    "saved": "Past tense and past participle of save, meaning to keep safe or rescue from harm or danger.",
    "sense": "A faculty by which the body perceives an external stimulus; one of the faculties of sight, smell, hearing, taste, and touch.",
    "serve": "To perform duties or services for (another person or an organization).",
    "sight": "The faculty or power of seeing.",
    "silly": "Having or showing a lack of common sense or judgment; absurd and foolish.",
    "since": "In the intervening period between (the time mentioned) and the time under consideration, typically the present.",
    "slate": "A fine-grained gray, green, or bluish metamorphic rock easily split into smooth, flat pieces.",
    "slice": "A thin, broad piece of food, such as bread, meat, or cake, cut from a larger portion.",
    "sorry": "Feeling distress, especially through sympathy with someone else's misfortune.",
    "sound": "Vibrations that travel through the air or another medium and can be heard when they reach a person's or animal's ear.",
    "south": "The direction or point on the mariner's compass at 180°, corresponding to the southward cardinal direction.",
    "suite": "A set of rooms designated for one person's or family's use or for a particular purpose.",
    "sunny": "Bright with sunlight.",
    "taste": "The sensation of flavor perceived in the mouth and throat on contact with a substance.",
    "teach": "To impart knowledge to or instruct (someone) as to how to do something.",
    "tense": "Stretched tight or rigid.",
    "there": "In, at, or to that place or position.",
    "those": "Used to identify specific people or things observed by the speaker at some distance.",
    "tight": "Fixed, fastened, or closed firmly; hard to move, undo, or open.",
    "tired": "In need of sleep or rest; weary.",
    "toast": "Sliced bread browned on both sides by exposure to radiant heat.",
    "touch": "To come into or be in contact with.",
    "tough": "Strong enough to withstand adverse conditions or rough or careless handling.",
    "tower": "A tall narrow building, either free-standing or forming part of a building such as a church or castle.",
    "trace": "Find or discover by investigation.",
    "track": "A rough path or road, typically one beaten by use rather than constructed.",
    "trade": "The action of buying and selling goods and services.",
    "train": "A series of connected railroad cars pulled or pushed by one or more locomotives.",
    "treat": "To behave toward or deal with in a certain way.",
    "trick": "A cunning or skillful act or scheme intended to deceive or outwit someone.",
    "tried": "Past tense and past participle of try, meaning to make an attempt or effort to do something.",
    "tumor": "A swelling of a part of the body, generally without inflammation, caused by an abnormal growth of tissue, whether benign or malignant.",
    "union": "The action or fact of joining or being joined, especially in a political context.",
    "vince": "A male given name, often a short form of Vincent.",
    "vocal": "Relating to the human voice.",
    "waste": "To use or expend carelessly, extravagantly, or to no purpose.",
    "watch": "To look at or observe attentively, typically over a period of time.",
    "water": "A colorless, transparent, odorless liquid that forms the seas, lakes, rivers, and rain and is the basis of the fluids of living organisms.",
    "waved": "Past tense and past participle of wave, meaning to move one's hand to and fro in greeting or farewell.",
    "where": "In or to what place or position.",
    "whose": "Belonging to or associated with which person.",
    "wired": "Having wires or electrical connections.",
    "woman": "An adult human female.",
    "worry": "To feel or cause to feel anxious or troubled about actual or potential problems.",
    "worse": "Of poorer quality or lower standard; less good or desirable.",
    "worth": "The level at which someone or something deserves to be valued or rated.",
    "would": "Expressing the conditional mood.",
    "wound": "An injury to living tissue caused by a cut, blow, or other impact, typically one in which the skin is cut or broken.",
    "yield": "To produce or provide (a natural, agricultural, or industrial product).",
    "youth": "The period between childhood and adult age.",
}
