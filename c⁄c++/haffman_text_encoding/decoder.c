#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

struct unit {
    char symb;
    struct unit *son[2];
};

void deltree(struct unit *root) {
    if (root->son[0] == NULL) free(root);
    else {
        deltree(root->son[0]);
        deltree(root->son[1]);
        free(root);
    }
}

struct unit *gettree(int code) {
    struct unit *tree = malloc(10);
    tree->son[0] = NULL;
    tree->son[1] = NULL;
    struct unit *currun;
    char c, c1 = 0;
    while (read(code, &c, 1) == 1) {
        currun = tree;
        read(code, &c1, 1);
        while ((read(code, &c1, 1) == 1) && (c1 != '\n')) {
            if (currun->son[c1 - '0'] == NULL) currun->son[c1 - '0'] = malloc(8);
            currun = currun->son[c1 - '0'];
        }
        currun->symb = c;
    }
    return tree;
}

//working but horrible old code
//argv[1] - ciphered file name; argv[2] - cipher file name; argv[3] - output file name;
int main(int argc, char *argv[]) {
    if (argc != 4) {
        if (argc == 2) {
            printf("First arg - ciphered file name, must exist\n");
            printf("Second arg - cipher file name, must exist\n");
            printf("Third arg - output file name\n");
            printf("Works correctly with text encoded by encoder.c\n");
            return 0;
        }
        printf("Wrong number of arguments. Run with parameter '-h' for help\n");
        return 0;
    }

    int cifertext = open(argv[1], O_RDONLY);
    if (cifertext < 0) {
        printf("Error opening %s \n", argv[1]);
        return 0;
    }

    int code = open(argv[2], O_RDONLY);
    if (code < 0) {
        printf("Error opening %s \n", argv[2]);
        return 0;
    }

    int output = open(argv[3], O_CREAT | O_WRONLY | O_TRUNC, -1);
    struct unit *dectree = gettree(code);
    struct unit *currun = dectree;
    unsigned char c, c1, c2, mask = (char) (1 << 7);
    int i, flag;

    read(cifertext, &c, 1);
    read(cifertext, &c1, 1);
    while (read(cifertext, &c2, 1) == 1) {
        for (i = 0; i < 7; i++) {
            if ((c & mask) != 0) flag = 1;
            else flag = 0;
            currun = currun->son[flag];
            if (currun->son[0] == NULL) {
                write(output, &(currun->symb), 1);
                currun = dectree;
            }
            c <<= 1;
        }
        c = c1;
        c1 = c2;
    }
    for (i = 0; i < (int) c1 - 1; i++) {
        if ((c & mask) != 0) flag = 1;
        else flag = 0;
        currun = currun->son[flag];
        if (currun->son[0] == NULL) {
            write(output, &(currun->symb), 1);
            currun = dectree;
        }
        c <<= 1;
    }

    deltree(dectree);
    close(cifertext);
    close(code);
    close(output);
    return 0;
}