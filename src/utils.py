import os
import sys
from src.exception import CustomException


def save_object(filepath,obj):
    try:
        obj.save(filepath)

    except Exception as e:
        raise CustomException(e,sys)