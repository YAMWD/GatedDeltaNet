CC ?= cc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -pedantic

all: gdn_eval gdn_attn_test

gdn_eval: gdn_eval.c gdn_model.c gdn_model.h
	$(CC) $(CFLAGS) -o $@ gdn_eval.c gdn_model.c -lm

gdn_attn_test: gdn_attn_test.c gdn_model.c gdn_model.h
	$(CC) $(CFLAGS) -o $@ gdn_attn_test.c gdn_model.c -lm

clean:
	rm -f gdn_eval gdn_attn_test
