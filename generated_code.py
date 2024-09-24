
def is_palindrome(s: str | int) -> bool:
    if not isinstance(s, (str, int)):
        raise TypeError("Input must be a string or an integer.")
    if isinstance(s, int):
        s = str(s)
    s = s.replace(" ", "").lower()
    s = ''.join(e for e in s if e.isalnum())
    return s == s[::-1]
