module util
    use config

    implicit none

    integer,external :: omp_get_max_threads

    contains
        
        subroutine parse_array_to_structure(data_in,data_out)
            implicit none

            real(8),intent(in) :: data_in(1:nwght)
            type(weights),intent(inout) :: data_out
            
            integer :: cntr,ii,jj
            
            cntr = 1
            
            !* hidden layer 1 
            do ii=0,D,1
                do jj=1,net_dim%hl1,1
                    data_out%hl1(jj,ii) = data_in(cntr) 
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
            
            !* hidden layer 2
            do ii=0,net_dim%hl1,1
                do jj=1,net_dim%hl2,1
                    !* include bias in weights
                    data_out%hl2(jj,ii) = data_in(cntr)
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
       
            !* final output weights
            do ii=0,net_dim%hl2,1
                !* include bias in weights
                data_out%hl3(ii) = data_in(cntr)
                cntr = cntr + 1
            end do
        end subroutine parse_array_to_structure

        subroutine parse_structure_to_array(data_in,data_out)
            implicit none

            type(weights),intent(in) :: data_in
            real(8),intent(inout) :: data_out(1:nwght)
            
            integer :: cntr,ii,jj
            cntr = 1
            
            !* hidden layer 1 
            do ii=0,D,1
                do jj=1,net_dim%hl1,1
                    data_out(cntr) = data_in%hl1(jj,ii)
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
            
            !* hidden layer 2
            do ii=0,net_dim%hl1,1
                do jj=1,net_dim%hl2,1
                    !* include bias in weights
                    data_out(cntr) = data_in%hl2(jj,ii)
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
       
            !* final output weights
            do ii=0,net_dim%hl2,1
                !* include bias in weights
                data_out(cntr) = data_in%hl3(ii)
                cntr = cntr + 1
            end do
        end subroutine parse_structure_to_array

        subroutine copy_weights_to_nobiasT()
            implicit none

            integer :: ii,jj

            do ii=1,net_dim%hl1
                do jj=1,D
                    net_weights_nobiasT%hl1(jj,ii) = net_weights%hl1(ii,jj)
                end do
            end do
            do ii=1,net_dim%hl2
                do jj=1,net_dim%hl1
                    net_weights_nobiasT%hl2(jj,ii) = net_weights%hl2(ii,jj)
                end do
            end do
        end subroutine copy_weights_to_nobiasT

        integer function total_num_weights()
            implicit none
        
            !* scratch 
            integer :: l1,l2,l3

            l1 = (D+1)*net_dim%hl1 
            l2 = (net_dim%hl1+1)*net_dim%hl2
            l3 = net_dim%hl2 + 1

            total_num_weights = l1+l2+l3
        end function total_num_weights

        subroutine allocate_dydx(set_type,conf)
            implicit none

            integer,intent(in) :: set_type,conf

            allocate(dydx(D,data_sets(set_type)%configs(conf)%n))
        end subroutine

        subroutine zero_weights(weights_out)
            implicit none

            type(weights),intent(inout) :: weights_out

            weights_out%hl1 = 0.0d0
            weights_out%hl2 = 0.0d0
            weights_out%hl3 = 0.0d0
        end subroutine zero_weights

        integer function num_threads()
            implicit none

            num_threads = omp_get_max_threads()
        end function num_threads

        logical function array_equal(arr1,arr2,ftol,rtol,verbose)
            implicit none

            real(8),intent(in) :: arr1(:),arr2(:),ftol,rtol
            logical,intent(in) :: verbose

            logical :: equal,tmp
            integer :: ii

            equal = .false.

            if (size(arr1).eq.size(arr2)) then
                tmp = .true.
                do ii=1,size(arr1)
                    if (scalar_equal(arr1(ii),arr2(ii),ftol,rtol,verbose).neqv..true.) then
                        tmp = .false.
                    end if
                end do
                if (tmp) then
                    equal = .true.
                end if
            end if
            array_equal = equal
        end function array_equal

        logical function scalar_equal(scl1,scl2,ftol,rtol,verbose)
            implicit none

            real(8),intent(in) :: scl1,scl2,ftol,rtol
            logical,intent(in) :: verbose

            logical :: equal

            equal = .false.

            if ( (abs(scl1-0.0d0).lt.1e-10).or.(abs(scl2-0.0d0).lt.1e-10) ) then
                !* use absolute difference
                if (abs(scl1-scl2).le.rtol) then
                    equal = .true.
                end if
            else
                if (abs(0.5d0*(scl1/scl2 + scl2/scl1) - 1.0d0).le.ftol) then
                    equal = .true.
                end if
            end if
            if ( (equal.neqv..true.).and.(verbose) ) then
                write(*,*) scl1,scl2,'are not equal'
            end if
            scalar_equal = equal
        end function scalar_equal

        logical function any_nan_oned(array)
            real(8),intent(in) :: array(:)

            integer :: dim(1:1),ii

            dim = shape(array)

            do ii=1,dim(1)
                if (isnan(array(ii))) then
                    any_nan_oned = .true.
                end if
            end do
            any_nan_oned = .false.
        end function any_nan_oned
        
        logical function any_nan_twod(array)
            real(8),intent(in) :: array(:,:)

            integer :: dim(1:2),ii,jj

            dim = shape(array)

            do jj=1,dim(2)
                do ii=1,dim(1)
                    if (isnan(array(ii,jj))) then
                        any_nan_twod = .true.
                    end if
                end do
            end do
            any_nan_twod = .false.
        end function any_nan_twod

        real(8) function get_config(set_type,conf,cell,atomic_number,positions,forces)
            implicit none

            integer,intent(in) :: set_type,conf
            real(8),intent(inout) :: cell(1:3,1:3),positions(:,:)
            real(8),intent(inout) :: atomic_number(:),forces(:,:)

            !* scratch
            integer :: atm

            cell(1:3,1:3) = data_sets(set_type)%configs(conf)%cell(1:3,1:3)
            
            do atm=1,data_sets(set_type)%configs(conf)%n,1
                forces(:,atm) = data_sets(set_type)%configs(conf)%current_fi(:,atm)
                positions(:,atm) = data_sets(set_type)%configs(conf)%r(:,atm)
                atomic_number(atm) = data_sets(set_type)%configs(conf)%z(atm)
            end do

            !* return total energy
            get_config = sum(data_sets(set_type)%configs(conf)%current_ei)
        end function get_config

        integer function get_nconf(set_type)
            implicit none

            integer,intent(in) :: set_type

            get_nconf = data_sets(set_type)%nconf
        end function get_nconf

        integer function get_natm(set_type,conf)
            implicit none

            integer,intent(in) :: set_type,conf

            get_natm = data_sets(set_type)%configs(conf)%n
        end function get_natm
end module util
