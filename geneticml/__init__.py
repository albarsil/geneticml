from functools import wraps
from flask import make_response, request, g, abort

from .parameters import BaseParameter
from .validators import Response

def _validate(parameter, kwargs) -> tuple:
    """
        Validate the arguments received by param with those ones defined on add

        inputs:
            - kwargs: A dictionary with key and value, which should match with the ones defined on add
        
        outputs:
            It returns a tuple with the status code of the request and the message
    """

    if parameter.key not in kwargs.keys():
        return 400, Response.MISSING_REQUIRED_PARAM.format(parameter.key)
        
    if (parameter.isrequired == True) & (kwargs[parameter.key] == None):
        return 400, Response.MISSING_REQUIRED_PARAM.format(parameter.key)

    value = kwargs[parameter.key]

    # Compare if the key is required or not
    if ((parameter.isrequired == False) & (value == None)):
        return 200, None
    
    # Compare the key types
    if isinstance(value, parameter.keytype) == False:
        return 400, Response.BAD_PARAM_TYPE.format(parameter.key, parameter.keytype, str(type(value)))

    # Compare the key bounduaries if was set
    if (parameter.min != None and parameter.max != None) & ((parameter.keytype == float) | (parameter.keytype == int)):
        if ((value > parameter.max) | (value < parameter.min)):
            return 400, Response.BAD_PARAM_BONDUARY.format(parameter.key, parameter.min, parameter.max, str(value))
    
    # Compare the key list elements
    if ((parameter.keytype == list) | (parameter.keytype == set) | (parameter.keytype == tuple)):

        # Compare the all the list elements have the type set
        elements = [x for x in value if type(x) != parameter.innertype]
        if len(elements) > 1:
            return 400, Response.BAD_PARAM_INNERTYPE.format(parameter.key, parameter.innertype, ",".join([str(x) for x in elements]))

        # Compare the list constrains if set
        if parameter.constrain != None:
            elements = [x for x in value if x not in parameter.constrain]
            if len(elements) > 0:
                return 400, Response.FAILED_CONSTRAINS_PARAM.format(parameter.key, ",".join(parameter.constrain), ",".join([str(x) for x in elements]))

        # Compare the list inner bonduaries if set
        if ((parameter.min != None) & (parameter.max != None)):
            elements = [x for x in value if ((x > parameter.max) | (x < parameter.min))]
            if len(elements) > 0:
                return 400, Response.BAD_PARAM_BONDUARY.format(parameter.key, parameter.min, parameter.max, ",".join([str(x) for x in elements]))

    return 200, None

def validate_param(key: str, keytype: str, isrequired: bool, innertype=None, maxmin: tuple=None, constrain: tuple=None):
    """
        Add a new parameter with the specified key and restrictions

        inputs:
            key: The parameter name
            keytype: The parameter type
            isrequired: True if the parameter is required else False
            innertype: If keytype is list, tuple or set, then you can use this argument to define the inner type of this collection
            maxmin: If one of keytype or innertype are int or float then you can use this argument to define a bonduary to it. Like (0,10)
            constrain: If one of keytype or innertype are str or boolean then you can use this argument to define a constrain list to their values

        outputs:
            void
    """

    if keytype in (list, tuple, set):
        if innertype == None:
            raise ValueError("If keytype is subtype of list, then you should set the innertype value")
        elif (maxmin != None) & (innertype not in (int, float)):
            raise ValueError("Max and min value should be set just if innertype is numeric")
        elif (constrain != None) & (innertype not in (str, bool, int)):
            raise ValueError("Constrain value should be set just if innertype is str, int or bool")
    else:
        if (maxmin != None) & (keytype not in (int, float)):
            raise ValueError("Max and min value should be set just if keytype is numeric")

        if (constrain != None) & (keytype not in (str, bool, int)):
            raise ValueError("Constrain value should be set just if keytype is str, int or bool")

    param = BaseParameter(
        key = key,
        keytype = keytype,
        innertype = innertype,
        isrequired = isrequired,
        max = maxmin[1] if maxmin != None else None,
        min = maxmin[0] if maxmin != None else None,
        constrain = constrain
    )

    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json(silent=True, force=False)

            if data is None:
                return make_response(Response.MISSING_JSON_BODY, 400)

            status_code, message = _validate(param, data)

            if status_code != 200:
                return make_response(message, status_code)

            g.data = data
            return f(*args, **kwargs)
        return decorated_function
    return decorator