#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

struct symbol {
    char symb;
    int freq;
    struct symbol *next;
    struct symbol *son0;
    struct symbol *son1;
};
struct string {
    char *s;
};

unsigned int sizeofstr(const char *a) {
    unsigned int i = 0;
    for (; a[i] != '\0'; i++);
    return i;
}

void deltree(struct symbol *root, int fcode, struct string *code) {
    if (root->son0 == NULL) {
        char *s = malloc(2);
        s[0] = root->symb;
        s[1] = ' ';
        write(fcode, s, 2);
        write(fcode, code[root->symb].s, sizeofstr(code[root->symb].s));
        s[0] = '\n';
        write(fcode, s, 1);
        free(s);
        free(root);
    } else {
        deltree(root->son0, fcode, code);
        deltree(root->son1, fcode, code);
        free(root);
    }
}

void setcode(struct symbol *root, char *s, struct string *code, const unsigned int size) {
    if (root->son0 == NULL) {
        code[root->symb].s = malloc(size + 2);
        sprintf(code[root->symb].s, "%s", s);
    } else {
        int sz = sizeofstr(s);
        s[sz] = '0';
        s[sz + 1] = '\0';
        setcode(root->son0, s, code, size);
        s[sz] = '1';
        s[sz + 1] = '\0';
        setcode(root->son1, s, code, size);
    }
}

int addstr(int output, char *s, int locked, int *c) {
    unsigned int i;
    for (i = 0; i < sizeofstr(s); i++) {
        if (locked == 0) {
            write(output, c, 1);
            *c = 0;
        }
        if (locked == -1) locked++;
        locked++;
        if (s[i] == '1') {
            int ci = 1;
            ci <<= 8 - locked;
            (*c) |= ci;
        }
        if (locked == 7) {
            (*c)++;
            locked = 0;
        }
    }
    return locked;
}

//working but horrible old code
//argv[1] - input file name; argv[2] - cipher text file name ; argv[3] - output file name
int main(int argc, char *argv[]) {
    if (argc != 4) {
        if (argc == 2) {
            printf("First arg - input file name, must exist\n");
            printf("Second arg - cipher file name\n");
            printf("Third arg - output file name\n");
            printf("Only symbols that fit into 8-bit char are encoded correctly\n");
            return 0;
        }
        printf("Wrong number of arguments. Run with parameter '-h' for help\n");
        return 0;
    }
    int freq[256];
    memset(freq, 0, 256 * sizeof(int));
    int input = open(argv[1], O_RDONLY);
    if (input < 0) {
        printf("Error opening %s \n", argv[1]);
        return 0;
    }
    char c;
    int c1 = 0;

    while (read(input, &c, 1) == 1) freq[c]++;
    lseek(input, 0, SEEK_SET);

    int i;
    unsigned int size = 0;
    struct symbol *root = NULL;
    struct symbol *curr = NULL;

    //making sorted list of symbols
    for (i = 0; i < 256; i++) {
        if (freq[i] != 0) {
            if (size == 0) {
                size++;
                root = malloc(sizeof(struct symbol));
                root->symb = (char) i;
                root->freq = freq[i];
                root->next = NULL;
                root->son0 = NULL;
                root->son1 = NULL;
            } else {
                size++;
                curr = root;
                if (curr->freq > freq[i]) {
                    struct symbol *curr1 = malloc(sizeof(struct symbol));
                    curr1->symb = (char) i;
                    curr1->freq = freq[i];
                    curr1->next = root;
                    curr1->son0 = NULL;
                    curr1->son1 = NULL;
                    root = curr1;
                } else {
                    while ((curr->next != NULL) && (curr->next->freq < freq[i])) curr = curr->next;
                    struct symbol *curr1 = malloc(30);
                    curr1->symb = (char) i;
                    curr1->next = curr->next;
                    curr1->freq = freq[i];
                    curr1->son0 = NULL;
                    curr1->son1 = NULL;
                    curr->next = curr1;
                }
            }
        }
    }
    if (size == 0) return -1;

    struct string code[256];
    for (i = 0; i < 256; i++) code[i].s = NULL;
    struct symbol *newun;
    while (root->next != NULL) {
        newun = malloc(30);
        newun->freq = root->freq + root->next->freq;
        newun->son0 = root;
        newun->son1 = root->next;
        newun->next = root->next->next;
        newun->symb = -1;
        root = newun;
        while ((newun->next != NULL) && (newun->next->freq < root->freq)) {
            newun = newun->next;
        }
        if (newun != root) {
            curr = root->next;
            root->next = newun->next;
            newun->next = root;
            root = curr;
        }
    }

    char *s = malloc(size + 2);
    s[0] = '\0';

    setcode(root, s, code, size);

    int output = open(argv[3], O_CREAT | O_WRONLY | O_TRUNC, -1);
    int locked = -1;
    while (read(input, &c, 1) == 1) {
        locked = addstr(output, code[c].s, locked, &c1);
    }
    write(output, &c1, 1);
    c1 = locked + 1;
    write(output, &c1, 1);

    int cdf = open(argv[2], O_CREAT | O_WRONLY | O_TRUNC, -1);
    deltree(root, cdf, code);

    for (i = 0; i < 256; i++) {
        if (code[i].s != 0) free(code[i].s);
    }

    close(cdf);
    close(input);
    close(output);
    return 0;
}