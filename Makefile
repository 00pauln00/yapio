yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2 -lpthread

ime : yapio.c
	mpicc -D YAPIO_IME -o yapio yapio.c -Wall -Wextra -O2 -lpthread \
		-L/opt/ddn/ime/lib -I/opt/ddn/ime/include -lim_client

asan :
	mpicc -g -o yapio yapio.c -Wall -Wextra -fsanitize=address -lpthread

debug : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O0 -g -ggdb -lpthread

clean :
	rm -f yapio.o yapio

test : yapio
	./yapio_tests.sh
