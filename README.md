В ветке polina переделка программы для решения задачи <<Применимость приближения однократного рассеяния при импульсном зондировании неоднородной  среды>>. Текущее состояние сломано, я начинал делать какие-то переделки, для того чтобы можно было использовать какие-то реалистичные значения коэффициентов. Код запускается, но в на экран отрисовывается какая-то ерунда.
В ветке master версия которая была использована на последней конференции.


Fork from cuda ratracing tutorial project https://github.com/straaljager/GPU-path-tracing-tutorial-3.

* Volume scettering was added.
* Phase point was extended with time coordinate to maintain nonstationry scene.
* Different branching methods was implemented. Main path tracing loop was replaced to stack emulation for this purpose.
* Sphere object definishion was reworked. Now they have separate structure to describe its inner media and boundary light source.

Setup:
* Build for x64
* Edit path to cuda example *.h in src
* Add path to cuda eample *.lib to additional lib dir in proj. settings
