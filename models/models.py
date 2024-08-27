import copy


models = {}


def register(name):#注册模型类
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):#创建模型实例
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])#先拷贝
        model_args.update(args)#再更新
    else:
        model_args = model_spec['args']

    model = models[model_spec['name']](**model_args)
    
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
