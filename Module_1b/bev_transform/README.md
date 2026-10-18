# BEV transform (Module 1b)

Два примера вида сверху:

| Скрипт | Идея |
|--------|------|
| `make_bev.py` | Поза из калиба `leftImage.yml` (`r`, `t`), область в метрах из `bev_config.json`, видео кадр за кадром |
| `birds_view.py` | Гомография по углам трапеции в кадре, врезка BEV в угол |

```bash
cd Module_1b/bev_transform

# калиб + JSON
python make_bev.py --config bev_config.json

# гомография без калиба
PYTHONPATH=../../srccam python birds_view.py
```

Управление `make_bev`: пробел — пауза, `n` — кадр, `q` — выход.  
Размеры области BEV — в `bev_config.json` (`near_m`, `far_m`, `half_width_m`, `bev_width_px`).
