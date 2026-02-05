"""Configuration for the capital factual recall task.

Template: "The capital of A is B." We ask the model to fill in B.
A can be a country, state, or province.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


# Default (region, capital) pairs: countries, US states, Canadian provinces.
# Format: (region_name, capital_name). Used for data generation and causal model pools.
DEFAULT_CAPITAL_PAIRS: List[Tuple[str, str]] = [
    # Countries - Europe
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("United Kingdom", "London"),
    ("Netherlands", "Amsterdam"),
    ("Switzerland", "Bern"),
    ("Sweden", "Stockholm"),
    ("Norway", "Oslo"),
    ("Poland", "Warsaw"),
    ("Greece", "Athens"),
    ("Portugal", "Lisbon"),
    ("Belgium", "Brussels"),
    ("Austria", "Vienna"),
    ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"),
    ("Ireland", "Dublin"),
    ("Czech Republic", "Prague"),
    ("Hungary", "Budapest"),
    ("Romania", "Bucharest"),
    # Countries - Asia
    ("Japan", "Tokyo"),
    ("China", "Beijing"),
    ("India", "New Delhi"),
    ("South Korea", "Seoul"),
    ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"),
    ("Indonesia", "Jakarta"),
    ("Malaysia", "Kuala Lumpur"),
    ("Singapore", "Singapore"),
    ("Philippines", "Manila"),
    ("Bangladesh", "Dhaka"),
    ("Pakistan", "Islamabad"),
    ("Saudi Arabia", "Riyadh"),
    ("United Arab Emirates", "Abu Dhabi"),
    ("Israel", "Jerusalem"),
    ("Turkey", "Ankara"),
    ("Iran", "Tehran"),
    ("Iraq", "Baghdad"),
    # Countries - Africa
    ("Egypt", "Cairo"),
    ("South Africa", "Cape Town"),
    ("Nigeria", "Abuja"),
    ("Kenya", "Nairobi"),
    ("Ethiopia", "Addis Ababa"),
    ("Ghana", "Accra"),
    ("Morocco", "Rabat"),
    ("Tanzania", "Dodoma"),
    ("Uganda", "Kampala"),
    ("Algeria", "Algiers"),
    # Countries - Americas
    ("Canada", "Ottawa"),
    ("Mexico", "Mexico City"),
    ("Brazil", "Brasília"),
    ("Argentina", "Buenos Aires"),
    ("Chile", "Santiago"),
    ("Colombia", "Bogotá"),
    ("Peru", "Lima"),
    ("Venezuela", "Caracas"),
    ("Ecuador", "Quito"),
    ("Uruguay", "Montevideo"),
    ("Cuba", "Havana"),
    ("Jamaica", "Kingston"),
    # Countries - Oceania
    ("Australia", "Canberra"),
    ("New Zealand", "Wellington"),
    ("Fiji", "Suva"),
    # Countries - Other
    ("Russia", "Moscow"),
    # US states (sample)
    ("California", "Sacramento"),
    ("Texas", "Austin"),
    ("New York", "Albany"),
    ("Florida", "Tallahassee"),
    ("Illinois", "Springfield"),
    ("Ohio", "Columbus"),
    ("Georgia", "Atlanta"),
    ("Michigan", "Lansing"),
    ("North Carolina", "Raleigh"),
    ("Virginia", "Richmond"),
    # Canadian provinces
    ("Ontario", "Toronto"),
    ("Quebec", "Quebec City"),
    ("British Columbia", "Victoria"),
    ("Alberta", "Edmonton"),
    ("Manitoba", "Winnipeg"),
]


@dataclass
class CapitalTaskConfig:
    """Configuration for the capital factual recall causal model."""

    # Prompt template: "The capital of {A} is " — model is asked to generate B.
    prompt_template: str = "The capital of {A} is "

    # List of (region, capital) pairs for sampling.
    capital_pairs: List[Tuple[str, str]] = None

    def __post_init__(self):
        if self.capital_pairs is None:
            self.capital_pairs = list(DEFAULT_CAPITAL_PAIRS)

    @property
    def regions(self) -> List[str]:
        """Unique region names (A)."""
        return list(dict.fromkeys(r for r, _ in self.capital_pairs))

    @property
    def capitals(self) -> List[str]:
        """Unique capital names (B)."""
        return list(dict.fromkeys(c for _, c in self.capital_pairs))

    def get_capital(self, region: str) -> str | None:
        """Return capital for region, or None if not in pairs."""
        d = dict(self.capital_pairs)
        return d.get(region)
