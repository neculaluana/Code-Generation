
def reverse_string(s):
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    return s[::-1].lower().replace(" ", "")
