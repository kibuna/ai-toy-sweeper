// 同梱の Chipmunk2D（chipmunk/、pymunk 7.3.1 の Munk2D）をビルドする。pymunk と同じ -std=c99、
// 最適化は Python の sysconfig の CFLAGS と同じ -O2 と -DNDEBUG（浮動小数点の結果を pymunk と同じにするため。
// -ffast-math は使わない）。
// CP_OVERRIDE_MESSAGE も pymunk と同じ: chipmunk.c の cpMessage（assert の報告）の定義を外す印で、他の計算は変わらない。
// 代わりに src/cp_message.c の cpMessage（Chipmunk の文に env の状況を足して stderr に書く）を同じフラグでビルドする。
fn main() {
    let mut b = cc::Build::new();
    b.include("chipmunk/include")
        .define("NDEBUG", None)
        .define("CP_OVERRIDE_MESSAGE", None)
        .flag("-std=c99")
        .flag("-fno-strict-overflow")
        .opt_level(2)
        .warnings(false);
    for e in std::fs::read_dir("chipmunk/src").unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_str().unwrap().to_string();
        // cpHastySpace（多スレッドの Space）は使わない
        if name.ends_with(".c") && name != "cpHastySpace.c" {
            b.file(&p);
        }
    }
    b.file("src/cp_message.c");
    b.compile("chipmunk");
    println!("cargo:rerun-if-changed=chipmunk");
    println!("cargo:rerun-if-changed=src/cp_message.c");
}
