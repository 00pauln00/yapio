yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2 -lpthread

asan :
	mpicc -g -o yapio yapio.c -Wall -Wextra -fsanitize=address -lpthread

clean :
	rm -f yapio.o yapio
