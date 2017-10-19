module util
    use config

    implicit none

    contains

        subroutine parse_weights_expand(flat_weights,nwght)
            !===================================================!
            ! update NN weights from a flat list of weights     !
            !===================================================!
            
            implicit none

            integer,intent(in) :: nwght
            real(8),intent(in) :: flat_weights(1:nwght)

            !* scratch
            integer :: ii,jj,cntr

            cntr = 1

            !* hidden layer 1
            do ii=1,net_dim%hl1,1
                do jj=1,D+1,1
                    net_weights%hl1(jj,ii) = flat_weights(cntr)
                    cntr = cntr + 1
                end do  !* loop features (and bias)
            end do !* loop linear models
            
            !* hidden layer 2
            do ii=1,net_dim%hl2,1
                do jj=1,net_dim%hl1+1,1
                    !* include bias in weights
                    net_weights%hl2(jj,ii) = flat_weights(cntr)
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
       
            !* final output weights
            do ii=1,net_dim%hl2+1,1
                !* include bias in weights
                net_weights%hl3(ii) = flat_weights(cntr)
                cntr = cntr + 1
            end do

        end subroutine
        
        subroutine parse_weights_flatten(flat_weights,nwght,deriv)
            implicit none

            integer,intent(in) :: nwght
            logical,intent(in) :: deriv
            real(8),intent(out) :: flat_weights(1:nwght)

            !* scratch
            integer :: ii,jj,cntr

            cntr = 1

            !* hidden layer 1 
            do ii=1,net_dim%hl1,1
                do jj=1,D+1,1
                    !* include bias in weights
                    if (deriv) then
                        flat_weights(cntr) = dydw%hl1(jj,ii)
                    else
                        flat_weights(cntr) = net_weights%hl1(jj,ii)
                    end if
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
            
            !* hidden layer 2
            do ii=1,net_dim%hl2,1
                do jj=1,net_dim%hl1+1,1
                    !* include bias in weights
                    if (deriv) then
                        flat_weights(cntr) = dydw%hl2(jj,ii)
                    else
                        flat_weights(cntr) = net_weights%hl2(jj,ii)
                    end if
                    cntr = cntr + 1
                end do !* loop features
            end do !* loop linear models
       
            !* final output weights
            do ii=1,net_dim%hl2+1,1
                !* include bias in weights
                if (deriv) then
                    flat_weights(cntr) = dydw%hl3(ii)
                else
                    flat_weights(cntr) = net_weights%hl3(ii)
                end if 
                cntr = cntr + 1
            end do
        end subroutine

        integer function total_num_weights()
            implicit none
        
            !* scratch 
            integer :: l1,l2,l3

            l1 = (D+1)*net_dim%hl1 
            l2 = (net_dim%hl1+1)*net_dim%hl2
            l3 = net_dim%hl2 + 1

            total_num_weights = l1+l2+l3
        end function

        logical function array_equal(arr1,arr2,ftol,rtol)
            implicit none

            real(8),intent(in) :: arr1(:),arr2(:),ftol,rtol

            logical :: equal,tmp
            integer :: ii

            equal = .false.

            if (size(arr1).eq.size(arr2)) then
                tmp = .true.
                do ii=1,size(arr1)
                    if (scalar_equal(arr1(ii),arr2(ii),ftol,rtol).neqv..true.) then
                        tmp = .false.
                    end if
                end do
                if (tmp) then
                    equal = .true.
                end if
            end if
            array_equal = equal
        end function array_equal

        logical function scalar_equal(scl1,scl2,ftol,rtol)
            implicit none

            real(8),intent(in) :: scl1,scl2,ftol,rtol

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
            scalar_equal = equal
        end function scalar_equal
end module util
