#!/usr/bin/env python
from src import train_imgcls, train_semseg
from src.utils.config import parse_config_and_args, convert_to_namespace, format_dict


if __name__ == '__main__':
    cfg, cfg_warnings = parse_config_and_args()
    cfg = convert_to_namespace(cfg)
    if len(cfg_warnings) > 0:
        print('\n'.join(cfg_warnings))
    print(format_dict(cfg.__dict__))
    {
        'imgcls': train_imgcls.main,
        'semseg': train_semseg.main,
    }[cfg.experiment](cfg)
