# Chipmunk2D（pymunk 7.3.1 同梱の Munk2D）

pymunk 7.3.1 の sdist（PyPI）の `Munk2D/` をそのまま写したもの（src/、include/、LICENSE.txt）。chipmunk_version
`2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3`。学習環境の Rust 版（native/）が、Python 版（pymunk）と同じ物理エンジンを
同じフラグ（`-std=c99`、`-O2`、`-DNDEBUG`、`CP_OVERRIDE_MESSAGE`）でビルドして使い、同じ軌跡になることを試験で確かめるため。
pymunk の版を上げたら、ここも写し直す（test/test_native_*.py が差を見つける）。

`CP_OVERRIDE_MESSAGE` は chipmunk.c の cpMessage（assert の報告）の定義を外す印で、計算は変わらない。pymunk は
pymunk_cffi/extensions.c の cpMessage（Python の logger に書いて戻る）に置き換え、native/ は native/src/cp_message.c の
cpMessage（chipmunk.c と同じ文を stderr に書き、続けて、どの env が reset / step の何をしていたかを書いて戻る）に
置き換える。どちらも止めるかは呼んだ側のマクロが決める: cpAssertHard は戻った後に abort する（pymunk も同じく落ちる）。
cpAssertSoft と cpAssertWarn は `-DNDEBUG` で消える（pymunk も Python の CFLAGS の `-DNDEBUG` で消える）。
