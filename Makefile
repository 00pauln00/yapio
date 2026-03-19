yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2 -lpthread

ime : yapio.c
	mpicc -D YAPIO_IME -o yapio yapio.c -Wall -Wextra -O2 -lpthread \
		-L/opt/ddn/ime/lib -I/opt/ddn/ime/include -lim_client

# ---------------------------------------------------------------------------
# Niova mode: direct IO to nisd, bypassing ublk, via libniova_block_client.
#
# Variable           Default         Description
# NIOVA_PATH         /usr/local      niova core install prefix
#                                    (headers at $(NIOVA_PATH)/include/niova/)
# NIOVA_BLOCK_PATH   /usr/local      niova-block install prefix
#                                    (headers at $(NIOVA_BLOCK_PATH)/include/niova/,
#                                     lib    at $(NIOVA_BLOCK_PATH)/lib/)
#
# Both default to /usr/local when installed together.  Override if they
# live at different prefixes:
#   make niova NIOVA_PATH=/opt/niova NIOVA_BLOCK_PATH=/opt/niova-block
#
# Library  : libniova_block_client  (client-only, same as niova-trivial-client)
# CFLAGS   : -DCLIENT_ONLY          (required by libniova_block_client headers)
# COMBINED : -lcurl -lxml2 -luring -lcjson  (mirrors Makefile.am COMBINED_LIBS)
# LDFLAGS  : -z noexecstack         (used on all niova binaries in Makefile.am)
# ---------------------------------------------------------------------------
NIOVA_PATH       ?= /usr/local
NIOVA_BLOCK_PATH ?= /usr/local

NIOVA_CFLAGS  = -DYAPIO_NIOVA -DCLIENT_ONLY \
                -I$(NIOVA_PATH)/include \
                -I$(NIOVA_BLOCK_PATH)/include

NIOVA_LIBS    = -L$(NIOVA_BLOCK_PATH)/lib \
                -lniova_block_client \
                -Wl,-rpath,$(NIOVA_BLOCK_PATH)/lib

# Mirrors COMBINED_LIBS in Makefile.am: $(CURL_LIBS) $(XML_LIBS) $(URING_LIBS) $(CJSON_LIBS)
COMBINED_LIBS = -lcurl -lxml2 -luring -lcjson -luuid

NIOVA_LDFLAGS = -Wl,-z,noexecstack

niova : yapio.c
	mpicc $(NIOVA_CFLAGS) \
		-o yapio yapio.c \
		-Wall -Wextra -O2 -lpthread \
		$(NIOVA_LIBS) $(COMBINED_LIBS) $(NIOVA_LDFLAGS)

niova-debug : yapio.c
	mpicc $(NIOVA_CFLAGS) \
		-o yapio yapio.c \
		-Wall -Wextra -O0 -g -ggdb -lpthread \
		$(NIOVA_LIBS) $(COMBINED_LIBS) $(NIOVA_LDFLAGS)

asan :
	mpicc -g -o yapio yapio.c -Wall -Wextra -fsanitize=address -lpthread

debug : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O0 -g -ggdb -lpthread

clean :
	rm -f yapio.o yapio

test : yapio
	./yapio_tests.sh
