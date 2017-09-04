yapio : yapio.c
	mpicc -o yapio yapio.c -Wall -Wextra -O2

asan :
	mpicc -o yapio yapio.c -Wall -Wextra -O2 -fsanitize=address

clean :
	rm -f yapio.o yapio
