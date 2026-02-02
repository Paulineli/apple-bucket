"""Core data structures for Entity Binding tasks.

This module provides the fundamental data structures needed to represent
entity binding tasks with arbitrary numbers of entity groups and entities per group.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from causalab.neural.token_position_builder import Template


def _build_conjoined_template(
    template: str,
    num_repetitions: int,
    delimiters: List[str],
    group_prefix: str = "g",
    capitalize_first: bool = True,
) -> str:
    """
    Build a conjoined template with unique variable names for each repetition.

    Takes a template like "{e0} loves {e1}" and creates:
    "{g0_e0} loves {g0_e1}, {g1_e0} loves {g1_e1}."

    This enables the declarative token position system to reference each
    variable by its unique group-qualified name.

    Args:
        template: Template string with {variable} placeholders, e.g., "{e0} loves {e1}"
        num_repetitions: Number of times to repeat the template
        delimiters: List of delimiters of length num_repetitions.
                    delimiters[i] is inserted after statement i.
        group_prefix: Prefix for group index (default "g"). Variables are
                      transformed as: "{e0}" -> "{g0_e0}"
        capitalize_first: Whether to capitalize first letter of first statement

    Returns:
        A new template string with unique variable names per repetition.

    Example:
        >>> _build_conjoined_template("{e0} loves {e1}", 2, [" and ", "."])
        "{g0_e0} loves {g0_e1} and {g1_e0} loves {g1_e1}."
    """
    if num_repetitions == 0:
        return ""

    if len(delimiters) != num_repetitions:
        raise ValueError(
            f"Expected {num_repetitions} delimiters, got {len(delimiters)}"
        )

    # Parse the template to find variables
    parsed = Template(template)

    result_parts = []

    for rep_idx in range(num_repetitions):
        # Build this repetition with group-qualified variable names
        for part_type, content in parsed.parts:
            if part_type == "literal":
                result_parts.append(content)
            else:  # variable
                # Transform variable name: "e0" -> "g0_e0"
                new_name = f"{group_prefix}{rep_idx}_{content}"
                result_parts.append("{" + new_name + "}")

        # Add delimiter after this repetition
        result_parts.append(delimiters[rep_idx])

    result = "".join(result_parts)

    # Capitalize first character if requested
    if capitalize_first and result:
        result = result[0].upper() + result[1:]

    return result


def _expand_delimiters(delimiters: List[str], num_statements: int) -> List[str]:
    """
    Expand FILL-style delimiter spec to a list of the correct length.

    Args:
        delimiters: Delimiter spec with "FILL" marker, e.g. [", ", "FILL", ", and ", "."]
        num_statements: Number of statements to join

    Returns:
        List of delimiters of length num_statements
    """
    fill_index = delimiters.index("FILL")
    filler = delimiters[fill_index - 1]
    result = delimiters[: fill_index - 1] + delimiters[fill_index + 1 :]

    while len(result) < num_statements:
        result.insert(fill_index - 1, filler)

    if len(result) > num_statements:
        result = result[-num_statements:]

    # For 2 statements, strip the comma from ", and " to get " and "
    if num_statements == 2 and ", and" in result[0]:
        result[0] = result[0].lstrip(",")

    return result


@dataclass
class EntityBindingTaskConfig:
    """
    Configuration for an entity binding task.

    This defines the structure of the task including:
    - Maximum dimensions (groups and entities per group)
    - Entity pools for each role position
    - Templates for generating text
    - Prompt formatting (prefix/suffix for instruction tuning)
    """

    max_groups: int  # Maximum number of entity groups (k)
    max_entities_per_group: int  # Maximum entities per group (d)
    entity_roles: Dict[int, str]  # {0: "person", 1: "food", 2: "location"}
    entity_pools: Dict[int, List[str]]  # {0: ["Pete", "Ann"], 1: ["jam", "pie"]}
    statement_template: str  # Template for the factual statements
    question_templates: Dict[
        Tuple[Tuple[int, ...], int], str
    ]  # Question templates by (query_indices, answer_index)
    delimiters: List[str]  # Delimiters used in statement conjunction
    prompt_prefix: str = ""  # Text to prepend before the prompt
    prompt_suffix: str = ""  # Text to append after the prompt
    statement_question_separator: str = " "  # Separator between statement and question
    fixed_query_indices: Optional[Tuple[int, ...]] = (
        None  # If set, always use these query indices
    )
    fixed_answer_index: Optional[int] = None  # If set, always use this answer index

    def build_mega_template(
        self,
        active_groups: int,
        query_indices: Tuple[int, ...],
        answer_index: int,
    ) -> str:
        """
        Build the mega template string for a given configuration.

        The mega template combines the conjoined statement template with the question template.
        Variable names in the result:
        - Statement entities: g0_e0, g0_e1, g1_e0, g1_e1, ...
        - Question entities: query_entity, person, food, object, location, ...

        Args:
            active_groups: Number of active groups
            query_indices: Tuple of entity indices being queried
            answer_index: Index of the answer entity

        Returns:
            Complete mega template string
        """
        delimiters = _expand_delimiters(self.delimiters, active_groups)
        statement_template_str = _build_conjoined_template(
            self.statement_template, active_groups, delimiters
        )
        question_template_str = self.question_templates.get(
            (query_indices, answer_index), ""
        )
        body = f"{statement_template_str}{self.statement_question_separator}{question_template_str}"
        return f"{self.prompt_prefix}{body}{self.prompt_suffix}"


class EntityGroup:
    """
    Represents one binding group G_i = (entity_0, entity_1, ..., entity_m).

    In the example "Pete loves jam", this would be EntityGroup(["Pete", "jam"], 0)
    """

    def __init__(self, entities: List[str], group_index: int):
        self.entities = entities
        self.group_index = group_index

    def get_entity(self, entity_index: int) -> Optional[str]:
        """Get entity at position entity_index, or None if index is out of bounds."""
        if 0 <= entity_index < len(self.entities):
            return self.entities[entity_index]
        return None

    def __repr__(self):
        return f"EntityGroup({self.entities}, group_{self.group_index})"


class BindingMatrix:
    """
    Represents the full binding matrix G with all entity groups.

    This is the core data structure that holds all the entity bindings
    for a particular instance of the task.
    """

    def __init__(
        self, groups: List[EntityGroup], max_groups: int, max_entities_per_group: int
    ):
        self.groups = groups
        self.active_groups = len(groups)
        self.max_groups = max_groups
        self.max_entities_per_group = max_entities_per_group

    def get_entity(self, group_idx: int, entity_idx: int) -> Optional[str]:
        """
        Get G[group_idx][entity_idx], return None if inactive/out of bounds.

        This handles the case where we have fewer active groups or entities
        than the maximum allowed by the configuration.
        """
        if group_idx < self.active_groups and group_idx < len(self.groups):
            return self.groups[group_idx].get_entity(entity_idx)
        return None

    def get_active_groups(self) -> int:
        """Get the maximum number of active groups."""
        return self.active_groups

    def get_entities_per_group(self) -> int:
        """Get the number of entities in the first active group (assumes all groups have same size)."""
        if self.groups:
            return len(self.groups[0].entities)
        return 0

    def __repr__(self):
        return f"BindingMatrix({self.groups}, active={self.active_groups})"


def create_sample_love_config() -> EntityBindingTaskConfig:
    """
    Create a sample configuration for the "love" task (Pete loves jam, Ann loves pie).

    This is a simple 2-entity-per-group task with people and foods.
    """
    return EntityBindingTaskConfig(
        max_groups=3,  # Support up to 3 groups
        max_entities_per_group=2,  # 2 entities per group (person, food)
        entity_roles={0: "person", 1: "food"},
        entity_pools={
            0: ["Pete", "Ann", "Tim", "Bob", "Sue", "Kate"],
            1: ["jam", "pie", "cake", "bread", "soup", "tea"],
        },
        statement_template="{e0} loves {e1}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # Query person (index 0), answer food (index 1)
            ((0,), 1): "What does {person} love?",
            # Query food (index 1), answer person (index 0)
            ((1,), 0): "Who loves {food}?",
        },
        prompt_prefix="We will ask a question about the following sentences.\n\n",
        prompt_suffix="\nAnswer:",
    )


def create_sample_action_config() -> EntityBindingTaskConfig:
    """
    Create a sample configuration for action tasks (Pete put jam in the cup).

    This is a 3-entity-per-group task with person, object, location.
    """
    return EntityBindingTaskConfig(
        max_groups=3,  # Support up to 3 action groups
        max_entities_per_group=3,  # 3 entities per group (person, object, location)
        entity_roles={0: "person", 1: "object", 2: "location"},
        entity_pools={
            0:  [
            "Pete", "Ann", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
            "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
            "Ben", "Liz", "Tom", "Joy"
            ],
            1: [
            "jam", "water", "book", "coin", "pen", "key", "phone", "watch",
            "cup", "box", "bag", "hat", "map", "card", "lamp", "ball",
            "rope", "tape", "tool", "clip"
            ],
            2: [
            "cup", "box", "table", "shelf", "drawer", "bag", "pocket", "basket",
            "desk", "chair", "floor", "rack", "case", "tray", "bin", "stand",
            "cabinet", "corner", "bench", "counter"
            ],
        },
        statement_template="{e0} put {e1} in the {e2}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # === SINGLE ENTITY QUERIES ===
            # Query person (0), answer object (1) - only mention the person
            ((0,), 1): "What did {person} put somewhere?",
            # Query person (0), answer location (2) - only mention the person
            ((0,), 2): "Where did {person} put something?",
            # Query object (1), answer person (0) - only mention the object
            ((1,), 0): "Who put {object} somewhere?",
            # Query object (1), answer location (2) - only mention the object
            ((1,), 2): "Where was {object} put?",
            # Query location (2), answer person (0) - only mention the location
            ((2,), 0): "Who put something in the {location}?",
            # Query location (2), answer object (1) - only mention the location
            ((2,), 1): "What was put in the {location}?",
            # === TWO ENTITY QUERIES ===
            # Query person+object (0,1), answer location (2) - mention both person and object
            ((0, 1), 2): "Where did {person} put {object}?",
            # Query person+location (0,2), answer object (1) - mention person and location
            ((0, 2), 1): "What did {person} put in the {location}?",
            # Query object+location (1,2), answer person (0) - mention object and location
            ((1, 2), 0): "Who put {object} in the {location}?",
        },
    )

def create_filling_liquids_config() -> EntityBindingTaskConfig:
    """
    Create a configuration for filling liquids tasks (John fills a cup with beer).

    This is a 3-entity-per-group task with person, container, liquid.
    Example: "John and Mary are working at a busy restaurant. To fulfill an order, 
    John fills a cup with beer and Mary fills a glass with wine. Who filled a cup with beer?"
    """
    return EntityBindingTaskConfig(
        max_groups=10,  # Support up to 10 groups
        max_entities_per_group=3,  # 3 entities per group (person, container, liquid)
        entity_roles={0: "person", 1: "container", 2: "liquid"},
        entity_pools={
            0: ["John", "Mary", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily", 
                "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
                "Ben", "Liz", "Tom", "Joy"],
            1: ["cup", "glass", "bottle", "mug", "jar", "pitcher", "bowl", "flask",
                "tumbler", "chalice", "vessel", "container", "tank", "can", "tube", "vial"],
            2: ["beer", "wine", "water", "juice", "milk", "coffee", "tea", "soda",
                "lemonade", "smoothie", "soup", "broth", "sauce", "syrup", "oil", "honey"],
        },
        statement_template="{e0} fills a {e1} with {e2}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # === SINGLE ENTITY QUERIES ===
            # Query person (0), answer container (1) - only mention the person
            ((0,), 1): "What container did {person} fill?",
            # Query person (0), answer liquid (2) - only mention the person
            ((0,), 2): "What did {person} fill a container with?",
            # Query container (1), answer person (0) - only mention the container
            ((1,), 0): "Who filled a {container}?",
            # Query container (1), answer liquid (2) - only mention the container
            ((1,), 2): "What was a {container} filled with?",
            # Query liquid (2), answer person (0) - only mention the liquid
            ((2,), 0): "Who filled something with {liquid}?",
            # Query liquid (2), answer container (1) - only mention the liquid
            ((2,), 1): "What container was filled with {liquid}?",
            # === TWO ENTITY QUERIES ===
            # Query person+container (0,1), answer liquid (2) - mention both person and container
            ((0, 1), 2): "What did {person} fill a {container} with?",
            # Query person+liquid (0,2), answer container (1) - mention person and liquid
            ((0, 2), 1): "What container did {person} fill with {liquid}?",
            # Query container+liquid (1,2), answer person (0) - mention container and liquid
            ((1, 2), 0): "Who filled a {container} with {liquid}?",
        },
    )


def create_music_config() -> EntityBindingTaskConfig:
    """
    Create a configuration for music tasks (John performed rock music on the piano).
    
    This is a 3-entity-per-group task with person, music/genre, instrument.
    Example: "At the music festival, John performed rock music on the piano, 
    and Mary performed pop music on the guitar. What music did Mary play on the guitar?"
    """
    return EntityBindingTaskConfig(
        max_groups=10,  # Support up to 10 groups
        max_entities_per_group=3,  # 3 entities per group (person, music, instrument)
        entity_roles={0: "person", 1: "music", 2: "instrument"},
        entity_pools={
            0: ["John", "Mary", "Bob", "Sue", "Tim", "Kate", "Dan", "Lily",
                "Max", "Eva", "Sam", "Zoe", "Leo", "Mia", "Noah", "Ava",
                "Ben", "Liz", "Tom", "Joy"],
            1: ["rock", "pop", "jazz", "classical", "blues", "country", "folk", "reggae",
                "hip hop", "electronic", "metal", "punk", "soul", "funk", "r&b", "gospel"],
            2: ["piano", "guitar", "drums", "violin", "flute", "saxophone", "trumpet", "bass",
                "cello", "clarinet", "harp", "organ", "ukulele", "banjo", "mandolin", "harmonica"],
        },
        statement_template="{e0} performed {e1} music on the {e2}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # === SINGLE ENTITY QUERIES ===
            # Query person (0), answer music (1) - only mention the person
            ((0,), 1): "What music did {person} perform?",
            # Query person (0), answer instrument (2) - only mention the person
            ((0,), 2): "What instrument did {person} perform on?",
            # Query music (1), answer person (0) - only mention the music
            ((1,), 0): "Who performed {music} music?",
            # Query music (1), answer instrument (2) - only mention the music
            ((1,), 2): "What instrument was {music} music performed on?",
            # Query instrument (2), answer person (0) - only mention the instrument
            ((2,), 0): "Who performed on the {instrument}?",
            # Query instrument (2), answer music (1) - only mention the instrument
            ((2,), 1): "What music was performed on the {instrument}?",
            # === TWO ENTITY QUERIES ===
            # Query person+instrument (0,2), answer music (1) - mention both person and instrument
            ((0, 2), 1): "What music did {person} play on the {instrument}?",
            # Query person+music (0,1), answer instrument (2) - mention person and music
            ((0, 1), 2): "What instrument did {person} perform {music} music on?",
            # Query music+instrument (1,2), answer person (0) - mention music and instrument
            ((1, 2), 0): "Who performed {music} music on the {instrument}?",
        },
    )


def create_boxes_config() -> EntityBindingTaskConfig:
    """
    Create a configuration for boxes tasks (The toy is in box B).
    
    This is a 2-entity-per-group task with object, box/container.
    Example: "The toy is in box B, and the medicine is in Box A. Which box is the medicine in?"
    """
    return EntityBindingTaskConfig(
        max_groups=10,  # Support up to 10 groups
        max_entities_per_group=2,  # 2 entities per group (object, box)
        entity_roles={0: "object", 1: "box"},
        entity_pools={
            0: ["toy", "medicine", "book", "coin", "pen", "key", "phone", "watch",
                "cup", "ball", "bag", "hat", "map", "card", "lamp", "rope",
                "tape", "tool", "clip", "pencil"],
            1: ["box A", "box B", "box C", "box D", "Box A", "Box B", "Box C", "Box D",
                "container A", "container B", "case A", "case B", "crate A", "crate B",
                "bin A", "bin B", "drawer A", "drawer B", "cabinet A", "cabinet B"],
        },
        statement_template="The {e0} is in {e1}",
        delimiters=[", ", "FILL", ", and ", "."],
        question_templates={
            # Query object (0), answer box (1) - only mention the object
            ((0,), 1): "Which box is the {object} in?",
            # Query box (1), answer object (0) - only mention the box
            ((1,), 0): "What is in {box}?",
        },
    )

