"""
Iris H20 - Native UTF-8 Byte-Level Tokenizer
============================================
Why Byte-Level?
1. Microscopic Vocabulary: Only 264 tokens (256 UTF-8 bytes + 8 special control tokens).
2. Universal Language Support: Seamlessly handles English, Sinhala, symbols, emojis, and code without needing external libraries like SentencePiece or HuggingFace Tokenizers.
3. Minimal VRAM: Embedding layer consumes less than 0.5 MB of GPU memory.
"""

class IrisByteTokenizer:
    # Special Control Tokens
    PAD_TOKEN = 256
    BOS_TOKEN = 257
    EOS_TOKEN = 258
    USER_TOKEN = 259
    IRIS_TOKEN = 260
    THINK_TOKEN = 261
    END_THINK_TOKEN = 262
    SEP_TOKEN = 263
    
    VOCAB_SIZE = 264

    SPECIAL_NAMES = {
        PAD_TOKEN: "<pad>",
        BOS_TOKEN: "<bos>",
        EOS_TOKEN: "<eos>",
        USER_TOKEN: "<user>",
        IRIS_TOKEN: "<iris>",
        THINK_TOKEN: "<think>",
        END_THINK_TOKEN: "</think>",
        SEP_TOKEN: "<sep>",
    }
    
    SPECIAL_LOOKUP = {v: k for k, v in SPECIAL_NAMES.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """
        Converts text with optional special tokens into a sequence of byte-level token IDs.
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.BOS_TOKEN)
            
        # Parse potential special tokens in text
        pos = 0
        while pos < len(text):
            found_special = False
            for tag, token_id in self.SPECIAL_LOOKUP.items():
                if text.startswith(tag, pos):
                    tokens.append(token_id)
                    pos += len(tag)
                    found_special = True
                    break
            if not found_special:
                # Encode char as utf-8 bytes
                char_bytes = text[pos].encode("utf-8")
                tokens.extend(list(char_bytes))
                pos += 1
                
        if add_special_tokens:
            tokens.append(self.EOS_TOKEN)
            
        return tokens

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a sequence of token IDs back into readable UTF-8 text.
        """
        out_str = ""
        byte_buffer = bytearray()
        
        for t in tokens:
            if t < 256:
                byte_buffer.append(t)
            else:
                # Flush existing bytes first
                if byte_buffer:
                    out_str += byte_buffer.decode("utf-8", errors="replace")
                    byte_buffer.clear()
                # Append special token string representation if desired
                if t in self.SPECIAL_NAMES and t not in (self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN):
                    out_str += self.SPECIAL_NAMES[t]
                    
        if byte_buffer:
            out_str += byte_buffer.decode("utf-8", errors="replace")
            
        return out_str


if __name__ == "__main__":
    tokenizer = IrisByteTokenizer()
    sample_text = "<user> Iris, how are you? ඔයාට කොහොමද? <iris> I am always here for you."
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)
    print("Sample Text :", sample_text)
    print("Tokens Count:", len(encoded))
    print("Decoded Text:", decoded)
    assert sample_text == decoded, "Encoding/Decoding roundtrip failed!"
    print("[SUCCESS] IrisByteTokenizer roundtrip test passed!")
