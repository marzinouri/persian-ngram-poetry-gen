# Persian Poetry Generator

## Overview

This project is a Persian poetry generator that focuses on the essential element of Persian poetry, "Qafiye" (rhyme). In Persian poetry, it's crucial that the endings of two lines, or "mesra," rhyme with each other, creating a harmonious and rhythmic structure. To achieve this, I employ a reversed n-gram model and G2P (grapheme into phoneme) conversion.

## How it Works

The Persian Poetry Generator follows these steps:

1. **Select a Rhyming Word:** The generator uses G2P (grapheme into phoneme) to choose a word that rhymes with the last word of one "mesra" (line of poetry). This is important in Persian poetry because certain vowel sounds are not written, and G2P helps identify rhyming words accurately.

2. **Reversed N-gram Generation:** After selecting the rhyming word, the generator reversely generates the other "mesra" starting from the last word and working its way back to the first one. This ensures that the two "mesras" rhyme seamlessly.

