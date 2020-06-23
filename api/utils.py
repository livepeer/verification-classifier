"""
Utility functions for API
"""
import flask.json as json
import os
import flask
from werkzeug.datastructures import FileStorage

def decode_form_to_json(request):
    """
    Parses JSON strings in Form fields to objects
    @param request:
    @return:
    """
    res = dict(request.form)
    for k, v in request.form.items():
        if isinstance(v, str):
            try:
                res[k] = json.loads(v)
            except:
                pass
    return res

def files_by_name(request):
    """
    Returns request files in dict with filename as a key. If file names are not unique, raises error.
    @param request:
    @return:
    """
    files = {}
    for key, fs in request.files.items():
        if fs.filename in files:
            raise Exception('File names are not unique')
        files[fs.filename] = fs
    return files

def save_file(filestorage: FileStorage, dir):
    """
    Saves FileStorage object to specified dir and returns its full name
    @param request:
    @return:
    """
    filename = os.path.join(dir, filestorage.filename)
    filestorage.save(filename)
    return filename