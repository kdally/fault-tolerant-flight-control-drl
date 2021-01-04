import random
import string


def get_ID(length):
    """
    generate new ID for controller
    """

    letters_and_digits = string.ascii_uppercase + string.digits
    result_str = ''.join((random.choice(letters_and_digits) for i in range(length)))
    return result_str
