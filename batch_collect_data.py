"""
batch_collect_data.py — Batch data collection for the LSTM pipeline.

Wraps collect_data.py to collect 30-frame sequences for multiple signs
in one session.

Usage:
    python batch_collect_data.py
"""

import os
import json

from collect_data import collect_data_for_sign


# Default vocabulary — 20 common ASL signs.
VOCABULARY = [
    "hello", "thank_you", "please", "sorry", "yes",
    "no", "help", "love", "family", "friend",
    "eat", "drink", "sleep", "work", "school",
    "home", "go", "stop", "want", "need",
]


def batch_collect_data(
    vocabulary,
    sequences_per_sign=30,
    output_dir="training_data",
):
    """
    Collect data for multiple signs in one session.

    For each sign, asks the user whether to collect, skip, or quit.

    Args:
        vocabulary:        List of sign names.
        sequences_per_sign: Sequences to collect per sign.
        output_dir:        Root directory for saved data.
    """
    print("=" * 50)
    print("  BATCH DATA COLLECTION")
    print("=" * 50)
    print(f"  Signs: {len(vocabulary)}")
    print(f"  Sequences per sign: {sequences_per_sign}")
    print()

    for idx, sign in enumerate(vocabulary):
        print(f"\n[{idx + 1}/{len(vocabulary)}]  Sign: '{sign}'")
        choice = input("  Collect? (yes / skip / quit): ").strip().lower()

        if choice in ("q", "quit"):
            print("Quitting batch collection.")
            break
        elif choice in ("s", "skip"):
            print(f"  Skipping '{sign}'.")
            continue
        else:
            collect_data_for_sign(
                sign_label=sign,
                sign_index=idx,
                num_sequences=sequences_per_sign,
                output_dir=output_dir,
            )

    # Save vocabulary mapping.
    vocab_path = os.path.join(output_dir, "vocabulary.json")
    vocab_dict = {str(i): sign for i, sign in enumerate(vocabulary)}
    os.makedirs(output_dir, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab_dict, f, indent=2)
    print(f"\nVocabulary saved to {vocab_path}")


def continue_previous_collection(output_dir="training_data"):
    """
    Resume a previous batch collection session.

    Detects which signs already have data and only collects for the rest.
    """
    if not os.path.exists(output_dir):
        print(f"No previous data found in '{output_dir}'.")
        print("Starting fresh collection instead.")
        batch_collect_data(VOCABULARY, output_dir=output_dir)
        return

    # Find already-collected signs.
    collected = set()
    for folder in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, folder)):
            parts = folder.split("_", 1)
            if len(parts) == 2:
                collected.add(parts[1])

    remaining = [s for s in VOCABULARY if s not in collected]

    if not remaining:
        print("All signs in the vocabulary have been collected!")
        return

    print(f"Already collected: {len(collected)} signs")
    print(f"Remaining: {len(remaining)} signs")
    print(f"  {remaining}")

    # Re-index remaining signs based on their position in VOCABULARY.
    for sign in remaining:
        idx = VOCABULARY.index(sign)
        collect_data_for_sign(
            sign_label=sign,
            sign_index=idx,
            num_sequences=30,
            output_dir=output_dir,
        )


def main():
    """Interactive menu."""
    print("=" * 50)
    print("  BATCH DATA COLLECTION MENU")
    print("=" * 50)
    print("  1. Start new batch collection (20 signs)")
    print("  2. Continue previous collection")
    print("  3. Custom vocabulary")
    print()

    choice = input("Choose (1/2/3): ").strip()

    if choice == "1":
        batch_collect_data(VOCABULARY)
    elif choice == "2":
        continue_previous_collection()
    elif choice == "3":
        raw = input("Enter signs separated by commas: ").strip()
        custom_vocab = [s.strip() for s in raw.split(",") if s.strip()]
        if custom_vocab:
            batch_collect_data(custom_vocab)
        else:
            print("No signs entered.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
