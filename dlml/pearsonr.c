
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <libgen.h>
#include <gsl/gsl_statistics.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>

void pearsonr(const double *x, const double *y, size_t n, double *r, double *p);

void pearsonr(const double *x, const double *y, size_t n, double *r, double *p) {
        double df, statistic;
        // compute the correlation coefficient
        *r = gsl_stats_correlation(x, 1, y, 1, n);
        // compute the p-value according to the following web page:
        // https://stats.stackexchange.com/questions/530191/why-gsl-p-value-for-pearson-and-spearman-are-different-from-python-or-r
        df = n - 2;
        statistic = sqrt(df) * (*r) / sqrt(1 - (*r) * (*r));
        *p = 2 * fmin(gsl_cdf_tdist_P(statistic, df), gsl_cdf_tdist_Q(statistic, df));
}

/*
static void print_array(const double *x, size_t n, const char *name) {
        int i;
        fprintf(stderr, "%s = np.array([", name);
        for (i=0; i<n; i++)
                fprintf(stderr, "%lf, ", x[i]);
        fprintf(stderr, "\b\b])\n");
}
*/

int main(int argc, char *argv[]) {
        int n = 100, i;
        double *x, *y;
        double R, p;
        if (argc > 1) {
                if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0) {
                        printf("usage: %s <n_samples>\n", basename(argv[0]));
                        exit(EXIT_SUCCESS);
                }
                n = atoi(argv[1]);
                if (n <= 1) {
                        fprintf(stderr, "Number of samples must be greater than 1.");
                        exit(1);
                }
        }
        x = (double *) malloc(n * sizeof(double));
        y = (double *) malloc(n * sizeof(double));
        srand(100);
        for (i=0; i<n; i++) {
                x[i] = (double) rand() / RAND_MAX;
                y[i] = (double) rand() / RAND_MAX;
        }
        //print_array(x, n, "x");
        //print_array(y, n, "y");
        pearsonr(x, y, n, &R, &p);
        printf("R = %g, p = %g\n", R, p);
        free(x);
        free(y);
        exit(EXIT_SUCCESS);
}

