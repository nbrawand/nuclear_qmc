import os


def get_new_file_name(file_name, postfix_int):
    """Return a new file name that doesn't exist in current directory.

    Parameters
    ----------
    file_name: str
        Proposed file name. If exists already then append `postfix_int` and check again. Return if doesn't exist.
    postfix_int: int
        Append to end of `file_name` if file already exists.

    Returns
    -------
    file_name: str
        A new file name that doesn't exist.

    """
    if os.path.isfile(file_name):
        new_int = postfix_int + 1
        file_name = file_name.replace(str(postfix_int), str(new_int))
        return get_new_file_name(file_name, new_int)
    else:
        return file_name
