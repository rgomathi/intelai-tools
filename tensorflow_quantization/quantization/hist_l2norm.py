import numpy as np

class HistMethods:
    def __init__(self, fp32_arr, bins, dst_nbins):
        self.fp32_arr  = fp32_arr
        self.bins = bins
        self.dst_nbins = dst_nbins

        self.histogram, _ = np.histogram(fp32_arr, self.bins)
        self.min_val = np.min(fp32_arr)
        self.max_val = np.max(fp32_arr)
        self.bin_width = (self.max_val - self.min_val) / self.bins
        print('min :', self.min_val)
        print('max :', self.max_val)
    
    def get_norm(self, delta_begin, delta_end, density):
            r"""
            Compute the norm of the values uniformaly distributed between
            delta_begin and delta_end.
            norm = density * (integral_{begin, end} x^2)
                 = density * (end^3 - begin^3) / 3
            """
            # assert norm_type == "L2", "Only L2 norms are currently supported"
            norm = 0.0
            # if norm_type == "L2":
            norm = (
                    delta_end * delta_end * delta_end
                    - delta_begin * delta_begin * delta_begin
                ) / 3

            # left_begin = np.min([0, delta_begin])
            # left_end = np.min([0, delta_end])
            # assert(left_begin * left_begin >= left_end * left_end)
            # norm += (left_begin * left_begin - left_end * left_end) / 2
        
            # right_begin = np.max([0, delta_begin])
            # right_end = np.max([0, delta_end])
            # assert(right_end * right_end >= right_begin * right_begin)
            # norm += (right_end * right_end - right_begin * right_begin) / 2
            
                # print('delta_begin :', delta_begin)
                # print('delta_end :', delta_end)
                # print('norm :', norm)
                # print('density :', density)
                # print('density*norm :', density*norm)
            return density * norm

    def compute_quantization_error(self, next_start_bin, next_end_bin):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            print('self.bin_width :', self.bin_width)

            dst_bin_width = self.bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            print('dst_bin_width :', dst_bin_width)

            if dst_bin_width == 0.0:
                return 0.0

            src_bin = np.arange(self.bins)
            # distances from the beginning of first dst_bin to the beginning and
            # end of src_bin
            src_bin_begin = (src_bin - next_start_bin) * self.bin_width
            src_bin_end = src_bin_begin + self.bin_width
            print('src_bin :', src_bin)
            print('src_bin_begin :', src_bin_begin)
            print('src_bin_end :', src_bin_end)

            # which dst_bins the beginning and end of src_bin belong to?
            dst_bin_of_begin = np.clip(src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width

            dst_bin_of_end = np.clip(src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)
            dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width
            print('dst_bin_of_begin :', dst_bin_of_begin)
            print('dst_bin_of_begin_center :', dst_bin_of_begin_center)
            print('dst_bin_of_end :', dst_bin_of_end)
            print('dst_bin_of_end_center :', dst_bin_of_end_center)

            density = self.histogram / self.bin_width
            # print('self.bin_width', self.bin_width)
            # print('density :', density)

            norm = np.zeros(self.bins)

            delta_begin = src_bin_begin - dst_bin_of_begin_center
            delta_end = dst_bin_width / 2
            print('delta_begin :', delta_begin)
            print('delta_end :', delta_end)

            norm += self.get_norm(delta_begin, delta_end, density)

            norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self.get_norm(
                -dst_bin_width / 2, dst_bin_width / 2, density)

            dst_bin_of_end_center = (
                dst_bin_of_end * dst_bin_width + dst_bin_width / 2
            )

            delta_begin = -dst_bin_width / 2
            delta_end = src_bin_end - dst_bin_of_end_center
            norm += self.get_norm(delta_begin, delta_end, density)

            return norm.sum()

    def hist_approx(self):
        r"""Non-linear parameter search.
        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
    
        #assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        # bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
        cSum = np.cumsum(self.histogram, axis=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")
        print('norm_min :', norm_min)
        print('total:', total)

        best_start_end_bins = {}
        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize
            print('alpha : {}, beta : {}, next_alpha : {}, next_beta : {}'.format(alpha, beta, next_alpha, next_beta))
            print('next_alpha * total :', next_alpha * total)

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            print('start_bin : {}, end_bin : {}, l : {}, r : {}'.format(start_bin, end_bin, l, r))

            #decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta
            # next_start_bin = l
            # next_end_bin = r
            # alpha = next_alpha
            # beta = next_beta


            print('next_start_bin :', next_start_bin)
            print('next_end_bin :', next_end_bin)

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self.compute_quantization_error(next_start_bin, next_end_bin)
            print('norm :', norm)

            if norm > norm_min:
                break

            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + self.bin_width * start_bin
        new_max = self.min_val + self.bin_width * (end_bin + 1)
        return new_min, new_max
    

    def compute_quantization_error_opt1(self, next_start_bin, next_end_bin):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            norm = 0
            print('self.bin_width :', self.bin_width)

            dst_bin_width = self.bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            print('dst_bin_width :', dst_bin_width)

            # if dst_bin_width == 0.0:
            #     return 0.0

            # src_bin = np.arange(self.bins)
            for src_bin in range(self.bins):
                #print('src_bin :', src_bin)
                # distances from the beginning of first dst_bin to the beginning and
                # end of src_bin
                src_bin_begin = (src_bin - next_start_bin) * self.bin_width
                src_bin_end = src_bin_begin + self.bin_width

                dst_bin_of_begin = np.min([np.max([np.floor((src_bin_begin ) / dst_bin_width), 0]), self.dst_nbins - 1])
                dst_bin_of_end = np.min([np.max([np.floor((src_bin_end) / dst_bin_width), 0]), self.dst_nbins - 1])

                dst_bin_of_begin_center = dst_bin_of_begin * dst_bin_width + dst_bin_width / 2

                density = self.histogram[src_bin] / self.bin_width

                delta_begin = src_bin_begin - dst_bin_of_begin_center

                if dst_bin_of_begin == dst_bin_of_end:
                    delta_end = src_bin_end - dst_bin_of_begin_center
                    norm += self.get_norm(delta_begin, delta_end, density)
                else:
                    delta_end = dst_bin_width / 2
                    norm += self.get_norm(delta_begin, delta_end, density)
                    norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self.get_norm(
                    -dst_bin_width / 2, dst_bin_width / 2, density)
                    dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
                    delta_begin = -dst_bin_width / 2
                    delta_end = src_bin_end - dst_bin_of_end_center
                    norm += self.get_norm(delta_begin, delta_end, density)
            
            print('norm : ', norm)
            return norm

    def hist_approx_opt1(self):
        r"""Non-linear parameter search.
        An approximation for L2 error minimization for selecting min/max.
        By selecting new min/max, we filter out outliers in input distribution.
        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
        caffe2/quantization/server/norm_minimization.cc
        """
    
        #assert self.histogram.size()[0] == self.bins, "bins mistmatch"
        # bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
        cSum = np.cumsum(self.histogram, axis=0)

        stepsize = 1e-5  # granularity
        alpha = 0.0  # lower bound
        beta = 1.0  # upper bound
        start_bin = 0
        end_bin = self.bins - 1
        norm_min = float("inf")
        print('norm_min :', norm_min)
        print('total:', total)

        best_start_end_bins = {}
        while alpha < beta:
            # Find the next step
            next_alpha = alpha + stepsize
            next_beta = beta - stepsize
            print('alpha : {}, beta : {}, next_alpha : {}, next_beta : {}'.format(alpha, beta, next_alpha, next_beta))
            print('next_alpha * total :', next_alpha * total)

            # find the left and right bins between the quantile bounds
            l = start_bin
            r = end_bin
            while l < end_bin and cSum[l] < next_alpha * total:
                l = l + 1
            while r > start_bin and cSum[r] > next_beta * total:
                r = r - 1

            print('start_bin : {}, end_bin : {}, l : {}, r : {}'.format(start_bin, end_bin, l, r))

            #decide the next move
            next_start_bin = start_bin
            next_end_bin = end_bin
            if (l - start_bin) > (end_bin - r):
                # move the start bin
                next_start_bin = l
                alpha = next_alpha
            else:
                # move the end bin
                next_end_bin = r
                beta = next_beta
            # next_start_bin = l
            # next_end_bin = r
            # alpha = next_alpha
            # beta = next_beta


            print('next_start_bin :', next_start_bin)
            print('next_end_bin :', next_end_bin)

            if next_start_bin == start_bin and next_end_bin == end_bin:
                continue

            # calculate the quantization error using next_start_bin and next_end_bin
            norm = self.compute_quantization_error_opt1(next_start_bin, next_end_bin)
            print('norm :', norm)

            if norm > norm_min:
                break

            norm_min = norm
            start_bin = next_start_bin
            end_bin = next_end_bin

        new_min = self.min_val + self.bin_width * start_bin
        new_max = self.min_val + self.bin_width * (end_bin + 1)
    

        #     if norm < norm_min:
        #         norm_min = norm
        #         best_start_end_bins[norm] = [start_bin, end_bin]
                    
        # print('best_start_end_bins :', best_start_end_bins)
        # best_start_bin = 0
        # norm_min = float("inf")
        # for norm in best_start_end_bins:
        #     [start_bin, end_bin] = best_start_end_bins[norm]
        #     # print('[0] is ', best_start_bins[nbins_selected][0])
        #     # print('[1] is ', norm)
        #     if norm < norm_min:
        #         norm_min = norm
        #         best_start_bin = start_bin
        #         best_end_bin = end_bin               
                

        # new_min = self.min_val + self.bin_width * best_start_bin
        # new_max = self.min_val + self.bin_width * (best_end_bin + 1)
    
        
        print('start_bin :', start_bin)
        print('end_bin :', end_bin)
        print('bin_width :', self.bin_width)
        print('self.bin_width * start_bin :', self.bin_width*start_bin)
        print('self.bin_width * end_bin +1 :', self.bin_width*end_bin+1)
        print('new_min :', new_min)
        print('new_max :', new_max)
        return new_min, new_max

    def compute_quantization_error_hist_brute(self, dst_bin_width, next_start_bin, norm):
            r"""
            Compute the quantization error if we use start_bin to end_bin as the
            min and max to do the quantization.
            """
            # bin_width = (self.max_val - self.min_val) / self.bins
            # print('self.bin_width :', self.bin_width)

            # dst_bin_width = self.bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
            # print('dst_bin_width :', dst_bin_width)

            # if dst_bin_width == 0.0:
            #     return 0.0

            # src_bin = np.arange(self.bins)
            for src_bin in range(self.bins):                
                # distances from the beginning of first dst_bin to the beginning and
                # end of src_bin
                src_bin_begin = (src_bin - next_start_bin) * self.bin_width
                src_bin_end = src_bin_begin + self.bin_width
                # print('src_bin :', src_bin)
                # print('src_bin_begin :', src_bin_begin)
                # print('src_bin_end :', src_bin_end)

                # which dst_bins the beginning and end of src_bin belong to?
                # dst_bin_of_begin = np.clip(src_bin_begin // dst_bin_width, 0, self.dst_nbins - 1)
                #dst_bin_of_end = np.clip(src_bin_end // dst_bin_width, 0, self.dst_nbins - 1)

                # dst_bin_of_begin = np.min([np.max([np.floor((src_bin_begin + 0.5 * dst_bin_width) // dst_bin_width), 0]), self.dst_nbins - 1])
                # dst_bin_of_end = np.min([np.max([np.floor((src_bin_end + 0.5 * dst_bin_width) // dst_bin_width), 0]), self.dst_nbins - 1])
                # dst_bin_of_begin_center = dst_bin_of_begin * dst_bin_width

                dst_bin_of_begin = np.min([np.max([np.floor((src_bin_begin) / dst_bin_width), 0]), self.dst_nbins - 1])
                dst_bin_of_end = np.min([np.max([np.floor((src_bin_end) / dst_bin_width), 0]), self.dst_nbins - 1])
                dst_bin_of_begin_center = dst_bin_of_begin * dst_bin_width + dst_bin_width/2

                # dst_bin_of_end_center = (dst_bin_of_end + 0.5) * dst_bin_width
                # print('dst_bin_of_begin :', dst_bin_of_begin)
                # print('dst_bin_of_begin_center :', dst_bin_of_begin_center)
                # print('dst_bin_of_end :', dst_bin_of_end)
                # print('dst_bin_of_end_center :', dst_bin_of_end_center)

                density = self.histogram[src_bin] / self.bin_width
                # print('self.bin_width', self.bin_width)
                # print('density :', density)

                # norm = np.zeros(self.bins)

                delta_begin = src_bin_begin - dst_bin_of_begin_center

                if dst_bin_of_begin == dst_bin_of_end:
                    delta_end = src_bin_end - dst_bin_of_begin_center
                    norm += self.get_norm(delta_begin, delta_end, density)
                else:
                    delta_end = dst_bin_width / 2
                    norm += self.get_norm(delta_begin, delta_end, density)

                    norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self.get_norm(
                    -dst_bin_width / 2, dst_bin_width / 2, density)
                    dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
                    delta_begin = -dst_bin_width / 2
                    delta_end = src_bin_end - dst_bin_of_end_center
                    norm += self.get_norm(delta_begin, delta_end, density)


            # print('delta_begin :', delta_begin)
            # print('delta_end :', delta_end)

            return norm

    def hist_brute(self):
        # bin_width = (self.max_val - self.min_val) / self.bins

        # cumulative sum
        total = sum(self.histogram)
        cSum = np.cumsum(self.histogram, axis=0)

        # stepsize = 1e-5  # granularity
        # alpha = 0.0  # lower bound
        # beta = 1.0  # upper bound
        # start_bin = 0
        # end_bin = self.bins - 1
        norm_min = float("inf")
        best_start_bin = -1

        print('norm_min :', norm_min)
        print('total:', total)
        best_nbins_selected = 1
        best_start_bins = {}

        for nbins_selected in range(1, self.bins):
            start_bin_begin = 0
            start_bin_end = self.bins - nbins_selected + 1
            # dst_bin_width = self.bin_width * nbins_selected / (self.dst_nbins - 1)
            dst_bin_width = self.bin_width * nbins_selected / (self.dst_nbins )

            for start_bin in range(start_bin_begin, start_bin_end):
                norm = 0
                # Go over each histogram bin and accumulate errors.

                #for src_bin in range(self.bins):
                norm = self.compute_quantization_error_hist_brute(dst_bin_width, start_bin, norm)

                if norm < norm_min:
                    norm_min = norm
                    best_start_bin = start_bin
                    # best_nbins_selected = nbins_selected

            best_start_bins[nbins_selected] = [best_start_bin, norm_min]
        
        print('best_start_bins :', best_start_bins)
        best_start_bin = 0
        norm_min = float("inf")
        for nbins_selected in range(1, self.bins):
            norm = best_start_bins[nbins_selected][1]
            # print('[0] is ', best_start_bins[nbins_selected][0])
            # print('[1] is ', norm)
            if norm < norm_min:
                norm_min = norm
                best_start_bin = best_start_bins[nbins_selected][0]
                
                best_nbins_selected = nbins_selected

        new_min = self.min_val + self.bin_width * best_start_bin
        new_max = self.min_val + self.bin_width * (best_start_bin + best_nbins_selected)
        print('new_min : {}, new_max : {}', new_min, new_max)
        return new_min, new_max
