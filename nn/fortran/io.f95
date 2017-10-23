module io
    implicit none

    contains
        subroutine error(routine,message)
            implicit none

            character(len=*),intent(in) :: routine,message

            write(*,*) '\n****************************'
            write(*,*) 'error raised in routine',routine,':'
            write(*,*) message
            write(*,*) '****************************\n'
            call exit(0)
        end subroutine

        subroutine unittest_header()
            implicit none

            write(*,*) ""
            write(*,*) "======================="
            write(*,*) "Running Unit Test Suite"
            write(*,*) "======================="
            write(*,*) ""
        end subroutine

        subroutine unittest_summary(tests)
            implicit none

            logical,intent(in) :: tests(:)

            if (all(tests)) then
                write(*,*) ""
                write(*,*) '---------------------------'
                write(*,*) "Unit test summary : SUCCESS"
                write(*,*) '---------------------------'
                write(*,*) ""
            else
                write(*,*) ""
                write(*,*) '---------------------------'
                write(*,*) "Unit test summary : FAILURE"
                write(*,*) '---------------------------'
                write(*,*) ""
            end if
        end subroutine

        subroutine unittest_test(num,success)
            implicit none

            integer,intent(in) :: num
            logical,intent(in) :: success

            if (success) then
                write(*,*) 'test     ',num,'    OK'
            else    
                write(*,*) 'test     ',num,'    FAILED'
            end if
        end subroutine unittest_test

        subroutine info_net()
            use config

            implicit none

            write(*,*) "NN layer    number of nodes"
            write(*,*) "1           ",net_dim%hl1
            write(*,*) "2           ",net_dim%hl2
            write(*,*)

            write(*,*) "Dimension of feature space : ",D
            write(*,*) "Nonlinear function         : ",nlf
            write(*,*)

            write(*,*) "weights layer     shape"
            write(*,*) "1                 ",shape(net_weights%hl1)
            write(*,*) "2                 ",shape(net_weights%hl2)
            write(*,*) "3                 ",shape(net_weights%hl3)
        end subroutine info_net
        
        subroutine info_set(set_type)
            use config

            implicit none

            integer,intent(in) :: set_type

            !* scratch 
            integer :: ii,jj
           
            if ( (set_type.lt.1).or.(set_type.gt.2) ) then
                call error("info_set","unsupported set_type. User error")
            end if

            write(*,*) "========="
            if (set_type.eq.1) then
                write(*,*) "train set"
            else
                write(*,*) "test set"
            end if
            write(*,*) "========="

            do ii=1,data_sets(set_type)%nconf,1
                write(*,*) ""
                write(*,*) "-------------"
                write(*,*) "structure ",ii
                write(*,*) "-------------"
                write(*,*) "features:"
                do jj=1,data_sets(set_type)%configs(ii)%n
                    write(*,*) data_sets(set_type)%configs(ii)%x(:,jj)
                end do
            
                write(*,*) ""
                write(*,*) 'energy : ',data_sets(set_type)%configs(ii)%energy
                write(*,*) "forces:"
                do jj=1,data_sets(set_type)%configs(ii)%n
                    write(*,*) data_sets(set_type)%configs(ii)%forces(:,jj)
                end do

            end do
        end subroutine info_set
end module io
