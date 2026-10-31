// Chipmunk の assert の報告（cpMessage）。build.rs が CP_OVERRIDE_MESSAGE を付けて chipmunk.c のものを外し、これを使う
// （pymunk 7.3.1 も同じ印で pymunk_cffi/extensions.c の cpMessage に置き換える）。
//
// pymunk と同じく、表示だけして戻る。止めるかは呼んだ側のマクロ（chipmunk.h）が決める:
// - cpAssertHard: 戻った後に abort()（NDEBUG でも残る。pymunk もここで落ちる）
// - cpAssertSoft（戻った後に abort()）と cpAssertWarn（続ける）: NDEBUG で消える（pymunk も Python の CFLAGS の
//   -DNDEBUG で消える）
// 表示は chipmunk.c のものと同じ文の後に、いま動いている env の状況（Rust の cp::toy_sweeper_cp_print_context）を足す。
// Python のプロセスが理由も場所も分からずに落ちないようにするため。
#include <stdarg.h>
#include <stdio.h>

#include "chipmunk/chipmunk.h"

void toy_sweeper_cp_print_context(void);

void cpMessage(const char *condition, const char *file, int line, int isError, int isHardError, const char *message, ...)
{
	(void)isHardError;
	fprintf(stderr, isError ? "Aborting due to Chipmunk error: " : "Chipmunk warning: ");
	va_list vargs;
	va_start(vargs, message);
	vfprintf(stderr, message, vargs);
	va_end(vargs);
	fprintf(stderr, "\n\tFailed condition: %s\n\tSource:%s:%d\n", condition, file, line);
	// Rust は stderr（fd 2）に直接書くので、順が入れ替わらないよう先に流す
	fflush(stderr);
	toy_sweeper_cp_print_context();
	fflush(stderr);
}
